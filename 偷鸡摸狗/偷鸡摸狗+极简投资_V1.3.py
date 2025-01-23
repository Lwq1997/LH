# 克隆自聚宽文章：https://www.joinquant.com/post/51937
# 标题：多策略5.2（修改模拟运行问题）
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
    g.rebalancing = 3  # 每个季度调仓一次
    g.month = 0 # 记录时间
    g.strategys_values = pd.DataFrame(
        columns=["s1", "s2", "s3"]
    )  # 子策略净值
    g.strategys_days = 250 # 取子策略净值最近250个交易日
    g.after_factor = [1, 1, 1]  # 后复权因子
    g.portfolio_value_proportion = [0, 0.5, 0.4, 0.1]

    # 创建策略实例
    set_subportfolios(
        [
            # 0号账户不做交易，仅资金进出，平衡仓位
            SubPortfolioConfig(
                context.portfolio.starting_cash * g.portfolio_value_proportion[i],
                "stock",
            )
            for i in range(4)
        ]
    )

    # 计算子策略净值、策略仓位动态调整
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
        run_monthly(simple_roa_adjust, 1, "9:33")


def process_initialize(context):

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

    simple_roa_strategy = Simple_ROA_Strategy(
        context,
        subportfolio_index=3,
        name="简单ROA策略",
    )

    g.strategys[jsg_strategy.name] = jsg_strategy
    g.strategys[all_day_strategy.name] = all_day_strategy
    g.strategys[simple_roa_strategy.name] = simple_roa_strategy


def jsg_prepare(context):
    g.strategys["搅屎棍策略"].prepare(context)


def jsg_check(context):
    g.strategys["搅屎棍策略"].check(context)


def jsg_adjust(context):
    g.strategys["搅屎棍策略"].adjust(context)


def all_day_adjust(context):
    g.strategys["全天候策略"].adjust(context)


def simple_roa_adjust(context):
    g.strategys["简单ROA策略"].adjust(context)


# 每日获取子策略净值
def get_strategys_values(context):
    df = g.strategys_values.copy()
    data = dict(
        zip(
            df.columns,
            [
                context.subportfolios[i + 1].total_value * g.after_factor[i]
                for i in range(len(df.columns))
            ],
        )
    )
    df.loc[len(df)] = data
    if len(df) > g.strategys_days:
        df = df.drop(0)
    g.strategys_values = df


import numpy as np
from scipy.optimize import minimize


# 计算最高夏普配比
def calculate_optimal_weights(context, alpha=0.5):
    print("目前仓位比例:")
    current_weights = [
        round(context.subportfolios[i].total_value / context.portfolio.total_value, 3)
        for i in range(len(g.portfolio_value_proportion))
    ]
    print(current_weights)
    df = g.strategys_values
    g.month += 1
    if len(df) < g.strategys_days or not g.month % g.rebalancing == 0:
        return

    # 计算每个策略的收益率
    returns = df.pct_change().dropna()

    # 计算每个策略的年化收益率
    annualized_returns = returns.mean() * 252

    # 计算协方差矩阵
    cov_matrix = returns.cov() * 252

    # 定义目标函数：负波动率调整后的夏普比率
    def negative_vasr(weights):
        portfolio_return = np.dot(weights, annualized_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - g.risk_free_rate) / portfolio_volatility
        vasr = sharpe_ratio / (1 + alpha * portfolio_volatility)
        return -vasr

    # 约束条件：权重之和为1
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        {"type": "ineq", "fun": lambda x: x - 0.05},  # 确保每个权重都大于等于0.05
    ]

    # 添加约束：每个策略前后配比之差不超过10%
    last_best_weights = g.portfolio_value_proportion[1:]  # 去掉第一个0
    constraints.append(
        {"type": "ineq", "fun": lambda x: 0.1 - np.abs(x - last_best_weights)}
    )

    # 添加约束：单个策略最大比重不超过50%
    constraints.append({"type": "ineq", "fun": lambda x: 0.5 - x})

    # 权重的初始猜测
    num_strategies = len(returns.columns)
    initial_weights = np.array([1 / num_strategies] * num_strategies)
    initial_weights = np.maximum(initial_weights, 0.05)  # 确保初始权重符合最低配比要求

    # 优化问题
    result = minimize(
        negative_vasr, initial_weights, method="SLSQP", constraints=constraints
    )

    # 输出最佳权重
    best_weights = result.x.tolist()
    g.portfolio_value_proportion = [0] + best_weights
    print("最佳权重:", [round(i, 3) for i in best_weights])


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
        value = (
            max(
                0,
                min(
                    context.portfolio.total_value
                    * g.portfolio_value_proportion[self.subportfolio_index]
                    - self.subportfolio.positions_value,
                    self.subportfolio.available_cash,
                ),
            )
            / count
        )

        for security in target:
            if security not in list(self.subportfolio.long_positions.keys()):
                self.order_target_value_(security, value)

    # 自定义下单
    def order_target_value_(self, security, value):
        current_data = get_current_data()
        if current_data[security].paused:
            log.info(security + ":今日停牌")
            return
        return order_target_value(security, value, pindex=self.subportfolio_index)

    # 计算后复权因子
    def get_net_values(self, amount):
        df = g.strategys_values
        if df.empty:
            return
        column_index = self.subportfolio_index - 1
        # 获取前一天的索引
        last_day_index = len(df) - 1

        # 获取前一天净值
        last_value = df.iloc[last_day_index, column_index]

        # 计算后复权因子, amount 代表分红金额
        g.after_factor[column_index] *= (last_value - amount) / last_value

    # 平衡账户间资金
    def balance_subportfolios(self, context):
        target = (
            g.portfolio_value_proportion[self.subportfolio_index]
            * context.portfolio.total_value
        )
        value = self.subportfolio.total_value
        # 仓位比例过高调出资金
        cash = self.subportfolio.transferable_cash  # 当前账户可取资金
        if cash > 0 and target < value:
            amount = min(value - target, cash)
            transfer_cash(
                from_pindex=self.subportfolio_index,
                to_pindex=0,
                cash=amount,
            )
            self.get_net_values(-amount)

        # 仓位比例过低调入资金
        cash = context.subportfolios[0].transferable_cash  # 0号账户可取资金
        if target > value and cash > 0:
            amount = min(target - value, cash)
            transfer_cash(
                from_pindex=0,
                to_pindex=self.subportfolio_index,
                cash=amount,
            )
            self.get_net_values(amount)

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
                if value - target > self.min_volume and minV < value - target:
                    self.order_target_value_(etf, target)
            self.balance_subportfolios(context)
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



# 简单ROA策略
class Simple_ROA_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)

        self.stock_sum = 1

    def filter(self, context):
        stocks = get_all_securities("stock", date = context.previous_date).index.tolist()
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
