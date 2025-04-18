# 克隆自聚宽文章：https://www.joinquant.com/post/51876
# 标题：多策略组合4.0（极致夏普）
# 作者：O_iX

# 导入函数库
from jqdata import *
from jqfactor import get_factor_values
import datetime
import math
from scipy.optimize import minimize


# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深300作为基准
    # set_benchmark("515080.XSHG")
    # 打开防未来函数
    set_option("avoid_future_data", True)
    # 开启动态复权模式(真实价格)
    set_option("use_real_price", True)
    # 输出内容到日志 log.info()
    log.info("初始函数开始运行且全局只运行一次")
    # 过滤掉order系列API产生的比error级别低的log
    log.set_level("order", "error")
    # 固定滑点设置ETF 0.001(即交易对手方一档价)
    set_slippage(FixedSlippage(0.02), type="stock")
    set_slippage(FixedSlippage(0.002), type="fund")
    # 股票交易总成本0.3%(含滑点)
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0.001,
            open_commission=0.0003,
            close_commission=0.0003,
            close_today_commission=0,
            min_commission=5,
        ),
        type="stock",
    )
    # 设置货币ETF交易佣金0
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0,
            close_commission=0,
            close_today_commission=0,
            min_commission=0,
        ),
        type="mmf",
    )
    # 全局变量
    g.strategys = {}
    g.risk_free_rate = 0.03  # 无风险收益率
    g.research_mode = False # 是否开启研究模式（不适用于回测）：关闭自动调仓，每个策略初始占比必须大于0
    g.strategys_values = pd.DataFrame(columns=["s1", "s2", "s3", "s4"])
    g.portfolio_value_proportion = [0, 0.5, 0.3, 0.15, 0.05]

    # 创建策略实例
    set_subportfolios(
        [
            # 0号账户不做交易，仅资金进出，平衡仓位
            SubPortfolioConfig(
                context.portfolio.starting_cash * g.portfolio_value_proportion[i],
                "stock",
            )
            for i in range(5)
        ]
    )

    jsg_strategy = JSG_Strategy(
        context,
        subportfolio_index=1,
        name="搅屎棍策略",
    )

    all_day_strategy = All_Day_Strategy(
        context,
        subportfolio_index=2,
        name="全天候策略",
    )

    rotation_etf_strategy = Rotation_ETF_Strategy(
        context,
        subportfolio_index=3,
        name="核心资产轮动策略",
    )

    simple_roa_strategy = Simple_ROA_Strategy(
        context,
        subportfolio_index=4,
        name="简单ROA策略",
    )

    g.strategys[jsg_strategy.name] = jsg_strategy
    g.strategys[all_day_strategy.name] = all_day_strategy
    g.strategys[rotation_etf_strategy.name] = rotation_etf_strategy
    g.strategys[simple_roa_strategy.name] = simple_roa_strategy

    # 是否开启研究模式
    if g.research_mode:
        run_daily(get_strategys_values, "18:00")
        run_monthly(calculate_optimal_weights, 1, "19:00")

    # 子策略执行计划
    if g.portfolio_value_proportion[1] > 0:
        run_daily(jsg_prepare, "9:05")
        run_weekly(jsg_adjust, 1, "9:31")
        run_daily(jsg_check, "14:50")
    if g.portfolio_value_proportion[2] > 0:
        run_monthly(all_day_adjust, 1, "9:40")
    if g.portfolio_value_proportion[3] > 0:
        run_daily(rotation_etf_adjust, "9:32")
    if g.portfolio_value_proportion[4] > 0:
        run_monthly(simple_roa_adjust, 1, "9:33")


def jsg_prepare(context):
    g.strategys["搅屎棍策略"].prepare(context)


def jsg_check(context):
    g.strategys["搅屎棍策略"].check(context)


def jsg_adjust(context):
    g.strategys["搅屎棍策略"].adjust(context)


def all_day_adjust(context):
    g.strategys["全天候策略"].adjust(context)


def rotation_etf_adjust(context):
    g.strategys["核心资产轮动策略"].adjust(context)


def simple_roa_adjust(context):
    g.strategys["简单ROA策略"].adjust(context)

# 每日获取子策略净值
def get_strategys_values(context):
    df = g.strategys_values
    data = dict(
        zip(
            df.columns,
            [context.subportfolios[i + 1].total_value for i in range(len(df.columns))],
        )
    )
    df.loc[len(df)] = data
    if len(df) > 500:
        df = df.drop(0)


# 计算最佳权重以最大化夏普比率
def calculate_optimal_weights(context):
    df = g.strategys_values
    if len(df) < 500:
        return
    # 计算每个策略的收益率
    returns = df.pct_change().dropna()

    # 计算每个策略的年化收益率
    annualized_returns = returns.mean() * 252

    # 计算协方差矩阵
    cov_matrix = returns.cov() * 252

    # 定义目标函数：负夏普比率
    def negative_sharpe_ratio(weights):
        portfolio_return = np.dot(weights, annualized_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - g.risk_free_rate) / portfolio_volatility
        return -sharpe_ratio

    # 约束条件：权重之和为1
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    # 权重的初始猜测
    initial_weights = np.array([1 / len(returns.columns)] * len(returns.columns))

    # 优化问题
    result = minimize(
        negative_sharpe_ratio, initial_weights, method="SLSQP", constraints=constraints
    )

    # 输出最佳权重
    best_weights = result.x
    print(best_weights)


# 策略基类
class Strategy:

    def __init__(self, context, subportfolio_index, name):
        self.subportfolio_index = subportfolio_index
        self.name = name
        self.subportfolio = context.subportfolios[self.subportfolio_index]
        self.stock_sum = 1
        self.hold_list = []
        self.limit_up_list = []
        self.fill_stock = "511880.XSHG"

    # 准备今日所需数据
    def _prepare(self, context):
        # 获取已持有列表
        self.hold_list = list(
            context.subportfolios[self.subportfolio_index].long_positions.keys()
        )
        # 获取昨日涨停列表
        if self.hold_list != []:
            df = get_price(
                self.hold_list,
                end_date=context.previous_date,
                frequency="daily",
                fields=["close", "high_limit"],
                count=1,
                panel=False,
                fill_paused=False,
            )
            df = df[df["close"] == df["high_limit"]]
            self.limit_up_list = list(df.code)
        else:
            self.limit_up_list = []

    # 检查昨日涨停票
    def _check(self, context):
        if self.limit_up_list != []:
            current_data = get_current_data()
            # 对昨日涨停股票观察到尾盘如不涨停则提前卖出，如果涨停即使不在应买入列表仍暂时持有
            for stock in self.limit_up_list:
                if current_data[stock].last_price < current_data[stock].high_limit:
                    # log.info("[%s]涨停打开，卖出" % (stock))
                    self.order_target_value_(stock, 0)
                # else:
                #     log.info("[%s]涨停，继续持有" % (stock))

    # 调仓(等权购买输入列表中的标的)
    def _adjust(self, context, target):

        # 调仓卖出
        for security in self.hold_list:
            if (security not in target) and (security not in self.limit_up_list):
                self.order_target_value_(security, 0)

        # 调整子账户间资金
        self.balance_subportfolios(context)

        # 调仓买入
        count = len(set(target) - set(self.subportfolio.long_positions))
        if count == 0 or self.stock_sum <= len(self.subportfolio.long_positions):
            return
        value = self.subportfolio.available_cash / count
        for security in target:
            if security not in list(self.subportfolio.long_positions.keys()):
                self.order_target_value_(security, value)

    # 自定义下单
    def order_target_value_(self, security, value):
        return order_target_value(security, value, pindex=self.subportfolio_index)

    # 平衡账户间资金
    def balance_subportfolios(self, context):
        if g.research_mode:
            return
        target = (
            g.portfolio_value_proportion[self.subportfolio_index]
            * context.portfolio.total_value
        )
        value = self.subportfolio.total_value
        cash = self.subportfolio.transferable_cash  # 当前账户可取资金
        # 仓位比例过高调出资金
        if cash > 0 and target < value:
            transfer_cash(
                from_pindex=self.subportfolio_index,
                to_pindex=0,
                cash=min(value - target, cash),
            )
        cash = context.subportfolios[0].transferable_cash  # 0号账户可取资金
        # 仓位比例过低调入资金
        if target > value and cash > 0:
            transfer_cash(
                from_pindex=0,
                to_pindex=self.subportfolio_index,
                cash=min(target - value, cash),
            )

    # 基础过滤(过滤科创北交、ST、停牌、次新股)
    def filter_basic_stock(self, context, stock_list):

        current_data = get_current_data()
        return [
            stock
            for stock in stock_list
            if not current_data[stock].paused
            and not current_data[stock].is_st
            and "ST" not in current_data[stock].name
            and "*" not in current_data[stock].name
            and "退" not in current_data[stock].name
            and not (
                stock[0] == "4"
                or stock[0] == "8"
                or stock[:2] == "68"
                or stock[0] == "3"
            )
            and not context.previous_date - get_security_info(stock).start_date
            < datetime.timedelta(375)
        ]

    # 过滤当前时间涨跌停的股票
    def filter_limitup_limitdown_stock(self, context, stock_list):
        current_data = get_current_data()
        return [
            stock
            for stock in stock_list
            if stock in self.subportfolio.long_positions
            or (
                current_data[stock].last_price < current_data[stock].high_limit
                and current_data[stock].last_price > current_data[stock].low_limit
            )
        ]

    # 判断今天是在空仓月
    def is_empty_month(self, context):
        # 根据g.pass_month跳过指定月份
        month = context.current_dt.month
        if month in self.pass_months:
            return True
        else:
            return False


# 搅屎棍策略
class JSG_Strategy(Strategy):

    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)

        self.stock_sum = 6
        # 判断买卖点的行业数量
        self.num = 1
        # 空仓的月份
        self.pass_months = [1, 4]

    def getStockIndustry(self, stocks):
        industry = get_industry(stocks)
        dict = {
            stock: info["sw_l1"]["industry_name"]
            for stock, info in industry.items()
            if "sw_l1" in info
        }
        return pd.Series(dict)

    # 获取市场宽度
    def get_market_breadth(self, context):
        # 指定日期防止未来数据
        yesterday = context.previous_date
        # 获取初始列表
        stocks = get_index_stocks("000985.XSHG")
        count = 1
        h = get_price(
            stocks,
            end_date=yesterday,
            frequency="1d",
            fields=["close"],
            count=count + 20,
            panel=False,
        )
        h["date"] = pd.DatetimeIndex(h.time).date
        df_close = h.pivot(index="code", columns="date", values="close").dropna(axis=0)
        # 计算20日均线
        df_ma20 = df_close.rolling(window=20, axis=1).mean().iloc[:, -count:]
        # 计算偏离程度
        df_bias = df_close.iloc[:, -count:] > df_ma20
        df_bias["industry_name"] = self.getStockIndustry(stocks)
        # 计算行业偏离比例
        df_ratio = (
            (df_bias.groupby("industry_name").sum() * 100.0)
            / df_bias.groupby("industry_name").count()
        ).round()
        # 获取偏离程度最高的行业
        top_values = df_ratio.loc[:, yesterday].nlargest(self.num)
        I = top_values.index.tolist()
        # log.info(
        #     [name for name in I],
        #     "  全市场宽度：",
        #     np.array(df_ratio.sum(axis=0).mean()),
        # )
        return I

    # 过滤股票
    def filter(self, context):
        stocks = get_index_stocks("399101.XSHE")
        stocks = self.filter_basic_stock(context, stocks)
        stocks = (
            get_fundamentals(
                query(
                    valuation.code,
                )
                .filter(
                    valuation.code.in_(stocks),
                    indicator.roa > 0,
                )
                .order_by(valuation.market_cap.asc())
            )
            .head(20)
            .code
        )
        stocks = self.filter_limitup_limitdown_stock(context, stocks)
        return stocks[: min(len(stocks), self.stock_sum)]

    # 择时
    def select(self, context):
        I = self.get_market_breadth(context)
        industries = {"银行I", "有色金属I", "煤炭I", "钢铁I", "采掘I"}
        if not industries.intersection(I) and not self.is_empty_month(context):
            L = self.filter(context)
        else:
            L = [self.fill_stock]
        return L

    ## 准备今日所需数据
    def prepare(self, context):
        self._prepare(context)

    ## 调仓
    def adjust(self, context):
        target = self.select(context)
        self._adjust(context, target)

    ## 检查昨日涨停票
    def check(self, context):
        self._check(context)


# 全天候ETF策略
class All_Day_Strategy(Strategy):

    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)

        # 最小交易额(限制手续费)
        self.min_volume = 2000
        # 全天候ETF组合参数
        self.etf_pool = [
            "511010.XSHG",  # 国债ETF
            "518880.XSHG",  # 黄金ETF
            "513100.XSHG",  # 纳指100
            # 2020年之后成立(注意回测时间)
            "515080.XSHG",  # 红利ETF
            "159980.XSHE",  # 有色ETF
            "162411.XSHE",  # 华宝油气LOF
            "159985.XSHE",  # 豆粕ETF
        ]
        # 标的仓位占比
        self.rates = [0.4, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05]

    ## 调仓
    def adjust(self, context):
        subportfolio = context.subportfolios[self.subportfolio_index]

        # 计算每个 ETF 的目标价值
        targets = {
            etf: subportfolio.total_value * rate
            for etf, rate in zip(self.etf_pool, self.rates)
        }
        # 首次开仓
        if not subportfolio.long_positions:
            for etf, target in targets.items():
                self.order_target_value_(etf, target)
        # 后续平衡仓位
        else:
            # 先卖出
            for etf, target in targets.items():
                value = subportfolio.long_positions[etf].value
                minV = subportfolio.long_positions[etf].price * 100
                if value - target > self.min_volume and minV > value - target:
                    self.order_target_value_(etf, target)

            # 后买入
            for etf, target in targets.items():
                value = subportfolio.long_positions[etf].value
                minV = subportfolio.long_positions[etf].price * 100
                if (
                    target - value > self.min_volume
                    and minV < subportfolio.available_cash
                    and minV < target - value
                ):
                    self.order_target_value_(etf, target)


# 核心资产轮动ETF策略
class Rotation_ETF_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)

        self.stock_sum = 1
        self.etf_pool = [
            "518880.XSHG",  # 黄金ETF（大宗商品）
            "513100.XSHG",  # 纳指100（海外资产）
            "159915.XSHE",  # 创业板100（成长股，科技股，中小盘）
            "510180.XSHG",  # 上证180（价值股，蓝筹股，中大盘）
        ]
        self.m_days = 25  # 动量参考天数

    # 打分
    def MOM(self, etf):
        # 获取历史数据
        df = attribute_history(etf, self.m_days, "1d", ["close"])
        y = np.log(df["close"].values)
        # 计算线性回归
        n = len(y)
        x = np.arange(n)
        weights = np.linspace(1, 2, n)  # 线性增加权重
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        # 计算年化收益率
        annualized_returns = math.pow(math.exp(slope), 250) - 1
        # 计算 R-squared
        residuals = y - (slope * x + intercept)
        r_squared = 1 - (
            np.sum(weights * residuals**2) / np.sum(weights * (y - np.mean(y)) ** 2)
        )
        # 返回动量得分
        return annualized_returns * r_squared

    # 择股
    def select(self):
        score_list = [self.MOM(etf) for etf in self.etf_pool]
        df = pd.DataFrame(index=self.etf_pool, data={"score": score_list})
        df = df.sort_values(by="score", ascending=False)
        df = df[(df["score"] > 0) & (df["score"] <= 5)]  # 安全区间，动量过高过低都不好
        target = df.index.tolist()
        if not target:
            target = [self.fill_stock]
        return target[: min(len(target), self.stock_sum)]

    # 调仓
    def adjust(self, context):
        target = self.select()
        self._prepare(context)
        self._adjust(context, target)


# 简单ROA策略
class Simple_ROA_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)

        self.stock_sum = 1

    def filter(self, context):
        stocks = get_all_securities("stock").index.tolist()
        stocks = self.filter_basic_stock(context, stocks)
        stocks = list(
            get_fundamentals(
                query(valuation.code, indicator.roa).filter(
                    valuation.code.in_(stocks),
                    valuation.pb_ratio > 0,
                    valuation.pb_ratio < 1,
                    indicator.adjusted_profit > 0,
                )
            )
            .sort_values(by="roa", ascending=False)
            .head(20)
            .code
        )
        stocks = self.filter_limitup_limitdown_stock(context, stocks)
        if not stocks:
            stocks = [self.fill_stock]
        return stocks[: self.stock_sum]

    def adjust(self, context):
        target = self.filter(context)
        self._prepare(context)
        self._adjust(context, target)
