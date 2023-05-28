import numpy as np
import pandas as pd
import pickle
from universal import tools


class PickleMixin(object):

    def save(self, filename):
        """ Save object as a pickle """
        with open(filename, 'wb') as f:
            pickle.dump(self, f, -1)

    @classmethod
    def load(cls, filename):
        """ Load pickled object. """
        with open(filename, 'rb') as f:
            return pickle.load(f)


class AlgoResult(PickleMixin):
    """  Fee is in a percentages as a one-round fee. """

    def __init__(self, X, B):
        """
        :param X: Price relatives.
        :param B: Weights.
        """
        # set initial values
        self._fee = 0.
        self._B = B
        self.rf_rate = 0.
        self._X = X
        self._recalculate()

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, _X):
        self._X = _X
        self._recalculate()

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, _B):
        self._B = _B
        self._recalculate()

    @property
    def fee(self):
        return self._fee

    @fee.setter
    def fee(self, value):
        """ Set transaction costs. Fees can be either float or Series
        of floats for individual assets with proper indices. """
        if isinstance(value, dict):
            value = pd.Series(value)
        if isinstance(value, pd.Series):
            missing = set(self.X.columns) - set(value.index)
            assert len(missing) == 0, 'Missing fees for {}'.format(missing)

        self._fee = value
        self._recalculate()

    def _recalculate(self):
        # calculate return for individual stocks
        r = (self.X - 1) * self.B
        self.asset_r = r + 1
        self.r = r.sum(axis=1) + 1

        # stock went bankrupt
        self.r[self.r < 0] = 0.

        # add fees
        if not isinstance(self._fee, float) or self._fee != 0:
            fees = (self.B.shift(-1).mul(self.r, axis=0) - self.B * self.X).abs()
            fees.iloc[0] = self.B.ix[0]
            fees.iloc[-1] = 0.
            fees *= self._fee

            self.asset_r -= fees
            self.r -= fees.sum(axis=1)

        self.r_log = np.log(self.r)

    @property
    def returns(self):
        return (self.X * self.B).sum(axis=1) + 1 - self.B.sum(axis=1)

    @property
    def weights(self):
        return self.B

    @property
    def equity(self):
        return self.r.cumprod()

    """bisection method to estimate transaction remained factor w"""
    def bisection(self, f, a, b, tol):
        if np.sign(f(a)) == np.sign(f(b)):
            raise Exception("The scalars a and b do not bound a root")
        m = (a + b) / 2

        if np.abs(f(m)) < tol:
            return m
        elif np.sign(f(a)) == np.sign(f(m)):
            return self.bisection(f, m, b, tol)
        elif np.sign(f(b)) == np.sign(f(m)):
            return self.bisection(f, a, m, tol)

    def update(self, last_b, curr_b, last_x, last_s):
        tilde_b = (last_b * last_x) / np.dot(last_b, last_x)
        hat_S = last_s * np.dot(last_b, last_x)

        f = lambda w: 1 - w - self._fee.iloc[0] * np.abs((tilde_b - curr_b * w))[1:].sum()
        w = self.bisection(f, 0, 1, 0.000001)
        desired_width = 900
        np.set_printoptions(linewidth=desired_width, threshold=np.inf, suppress=True, precision=4)
        print("Portfolio everyday :", list(last_b.values))
        print(f"{'Cumulative wealth everyday: '}{hat_S:.4f}")

        return hat_S * w

    @property
    def final_wealth(self):
        last_s = 1 - 1 * self._fee.iloc[0] * (self.B.iloc[0].sum(axis=0))
        for i in range(len(self.B)-1):
            last_s = self.update(self.B.iloc[i], self.B.iloc[i + 1], self.X.iloc[i], last_s)

        # calculate final wealth.
        print("Portfolio everyday:", list(self.B.iloc[-1]))
        final_s = last_s * np.dot(self.B.iloc[-1], self.X.iloc[-1])
        return f"{'Final cumulative wealth: '}{final_s:.4f}"

    @property
    def sharpe(self):
        """ Compute sharpe ratio.
        """
        mu = (self.r - 1).mean()
        sd = (self.r - 1).std()
        sh = (mu - self.rf_rate) / sd
        return sh

    @property
    def winning_pct(self):
        x = self.r_log
        win = (x > 0).sum()
        all_trades = (x != 0).sum()
        return float(win) / all_trades

    @property
    def volatility(self):
        return np.sqrt(self.freq()) * self.r.std()

    @property
    def max_drawdown(self):
        """ Returns highest drawdown in percentage. """
        x = self.equity
        return max(1. - x / x.cummax())

    def freq(self, x=None):
        """ Number of data items per day. If data does not contain
        datetime index, assume daily frequency with 330 trading minutes a day."""
        x = x or self.r
        return tools.freq(x.index)

    def summary(self, name=None):
        return f"""Summary{'' if name is None else ' for ' + name}:
    Sharpe ratio: {self.sharpe:.4f}
    Volatility: {self.volatility:.4%}
    Max drawdown: {self.max_drawdown: .4%}
    Winning ratio: {self.winning_pct: .4%}
        """


class ListResult(list, PickleMixin):
    """ List of AlgoResults. """

    def __init__(self, results=None, names=None):
        results = results if results is not None else []
        names = names if names is not None else []
        super(ListResult, self).__init__(results)
        self.names = names

    def to_dataframe(self):
        """ Calculate equities for all results and return one dataframe. """
        eq = {}
        for result, name in zip(self, self.names):
            eq[name] = result.equity
        return pd.DataFrame(eq)

    def summary(self):
        return '\n'.join([result.summary(name) for result, name in zip(self, self.names)])
