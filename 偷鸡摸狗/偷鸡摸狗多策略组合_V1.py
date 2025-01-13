# 克隆自聚宽文章：https://www.joinquant.com/post/51758
# 标题：多策略组合2.0（高手续费，近五年年化30%，回撤7%）
# 作者：O_iX

# 导入函数库
from jqdata import *
from jqfactor import get_factor_values
import datetime
import math


# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark("515080.XSHG")
    # 打开防未来函数
    set_option("avoid_future_data", True)
    # 开启动态复权模式(真实价格)
    set_option("use_real_price", True)
    # 输出内容到日志 log.info()
    log.info("初始函数开始运行且全局只运行一次")
    # 过滤掉order系列API产生的比error级别低的log
    log.set_level("order", "error")
    # 固定滑点设置股票0.01，基金0.001(即交易对手方一档价)
    set_slippage(FixedSlippage(0.02), type="stock")
    set_slippage(FixedSlippage(0.002), type="fund")
    # 设置股票交易印花税千一，佣金万三
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
    # 持久变量
    g.strategys = {}
    g.portfolio_value_proportion = [0, 0.3, 0.5, 0.2]

    # 创建策略实例
    set_subportfolios(
        [
            # 第一个子账户不做交易，用来平衡仓位
            SubPortfolioConfig(
                context.portfolio.starting_cash * g.portfolio_value_proportion[0],
                "stock",
            ),
            SubPortfolioConfig(
                context.portfolio.starting_cash * g.portfolio_value_proportion[1],
                "stock",
            ),
            SubPortfolioConfig(
                context.portfolio.starting_cash * g.portfolio_value_proportion[2],
                "stock",
            ),
            SubPortfolioConfig(
                context.portfolio.starting_cash * g.portfolio_value_proportion[3],
                "stock",
            ),
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
    g.strategys[jsg_strategy.name] = jsg_strategy
    g.strategys[all_day_strategy.name] = all_day_strategy
    g.strategys[rotation_etf_strategy.name] = rotation_etf_strategy

    # 定期平衡子账户资金
    run_monthly(balance_subportfolios, 1, "9:00")

    # 子策略执行计划
    if g.portfolio_value_proportion[1] > 0:
        run_daily(jsg_prepare, "9:05")
        run_weekly(jsg_adjust, 1, "9:31")
        run_daily(jsg_check, "14:50")
    if g.portfolio_value_proportion[2] > 0:
        run_monthly(all_day_adjust, 1, "9:40")
    if g.portfolio_value_proportion[3] > 0:
        run_daily(rotation_etf_adjust, "9:32")


# 平衡子账户资金
def balance_subportfolios(context):
    length = len(g.portfolio_value_proportion)
    print("每月平衡子账户仓位")
    # 计算平衡前仓位比例
    print(
        "调整前："
        + str(
            [
                context.subportfolios[i].total_value / context.portfolio.total_value
                for i in range(length)
            ]
        )
    )
    # 先把所有可用资金打入一号资金仓位
    for i in range(1, length):
        target = g.portfolio_value_proportion[i] * context.portfolio.total_value
        value = context.subportfolios[i].total_value
        if context.subportfolios[i].available_cash > 0 and target < value:
            transfer_cash(
                from_pindex=i,
                to_pindex=0,
                cash=min(value - target, context.subportfolios[i].available_cash),
            )
    # 如果子账户仓位过低，从一号仓位往其中打入资金
    for i in range(1, length):
        target = g.portfolio_value_proportion[i] * context.portfolio.total_value
        value = context.subportfolios[i].total_value
        if target > value and context.subportfolios[0].available_cash > 0:
            transfer_cash(
                from_pindex=0,
                to_pindex=i,
                cash=min(target - value, context.subportfolios[0].available_cash),
            )
    # 计算平衡后仓位比例
    print(
        "调整后："
        + str(
            [
                context.subportfolios[i].total_value / context.portfolio.total_value
                for i in range(length)
            ]
        )
    )


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


# 策略基类
class Strategy:

    def __init__(self, context, subportfolio_index, name):
        self.subportfolio_index = subportfolio_index
        self.name = name

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
                    log.info("[%s]涨停打开，卖出" % (stock))
                    self.close_position(stock)
                else:
                    log.info("[%s]涨停，继续持有" % (stock))

    # 调仓
    def _adjust(self, context, target):
        subportfolio = context.subportfolios[self.subportfolio_index]
        # 调仓卖出
        for security in self.hold_list:
            if (security not in target) and (security not in self.limit_up_list):
                self.close_position(security)
        position_count = len(subportfolio.long_positions)
        log.info('当前持仓--',subportfolio.long_positions,'--持仓总数--',position_count)
        # 调仓买入
        if len(target) > position_count:
            buy_num = min(len(target), self.stock_sum - position_count)
            value = subportfolio.available_cash / buy_num
            for security in target:
                if security not in list(subportfolio.long_positions.keys()):
                    if self.open_position(security, value):
                        if position_count == len(target):
                            break

    # 自定义下单
    def order_target_value_(self, security, value):
        if value == 0:
            log.debug("Selling out %s" % (security))
        else:
            log.debug("Order %s to value %f" % (security, value))
        return order_target_value(security, value, pindex=self.subportfolio_index)

    # 买单只
    def open_position(self, security, value):
        order = self.order_target_value_(security, value)
        if order != None and order.filled > 0:
            return True
        return False

    # 卖单只
    def close_position(self, security):
        order = self.order_target_value_(security, 0)  # 可能会因停牌失败
        if order != None:
            if order.status == OrderStatus.held and order.filled == order.amount:
                return True
        return False

    # 过滤停牌股票
    def filter_paused_stock(self, stock_list):
        current_data = get_current_data()
        return [stock for stock in stock_list if not current_data[stock].paused]

    # 过滤ST及其他具有退市标签的股票
    def filter_st_stock(self, stock_list):
        current_data = get_current_data()
        return [
            stock
            for stock in stock_list
            if not current_data[stock].is_st
               and "ST" not in current_data[stock].name
               and "*" not in current_data[stock].name
               and "退" not in current_data[stock].name
        ]

    # 过滤科创北交股票
    def filter_kcbj_stock(self, stock_list):
        for stock in stock_list[:]:
            if (
                    stock[0] == "4"
                    or stock[0] == "8"
                    or stock[:2] == "68"
                    or stock[0] == "3"
            ):
                stock_list.remove(stock)
        return stock_list

    # 过滤当前时间涨跌停的股票
    def filter_limitup_limitdown_stock(self, context, stock_list):
        subportfolio = context.subportfolios[self.subportfolio_index]
        current_data = get_current_data()
        return [
            stock
            for stock in stock_list
            if stock in subportfolio.long_positions.keys()
               or (
                       current_data[stock].last_price < current_data[stock].high_limit
                       and current_data[stock].last_price > current_data[stock].low_limit
               )
        ]

    # 过滤次新股
    def filter_new_stock(self, context, stock_list, days):
        yesterday = context.previous_date
        return [
            stock
            for stock in stock_list
            if not yesterday - get_security_info(stock).start_date
                   < datetime.timedelta(days)
        ]

    # 过滤股价高于10元的股票
    def filter_high_price_stock(self, stock_list):
        last_prices = history(
            1, unit="1m", field="close", security_list=stock_list
        ).iloc[0]
        return last_prices[last_prices < 10].index.tolist()


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
        print(
            [name for name in I],
            "  全市场宽度：",
            np.array(df_ratio.sum(axis=0).mean()),
        )
        return I

    # 过滤股票(请勿改变代码顺序)
    def filter(self, context):
        stocks = get_index_stocks("399101.XSHE", context.current_dt)
        stocks = self.filter_kcbj_stock(stocks)
        stocks = self.filter_st_stock(stocks)
        stocks = self.filter_new_stock(context, stocks, 375)
        stocks = self.filter_paused_stock(stocks)
        stocks = get_fundamentals(
            query(
                valuation.code,
            )
            .filter(
                valuation.code.in_(stocks),
                income.np_parent_company_owners > 0,  # 归属于母公司所有者的净利润大于0
                income.net_profit > 0,  # 净利润大于0
                income.operating_revenue > 1e8,  # 营业收入大于1亿
            )
            .order_by(valuation.market_cap.asc())
        )["code"].tolist()
        stocks = self.filter_limitup_limitdown_stock(context, stocks)
        select_stock = stocks[: min(len(stocks), self.stock_sum)]
        return select_stock


    #  判断今天是在空仓月
    def is_empty_month(self, context):
        # 根据g.pass_month跳过指定月份
        month = context.current_dt.month
        if month in self.pass_months:
            return True
        else:
            return False

    # 择时
    def select(self, context):
        I = self.get_market_breadth(context)
        industries = {"银行I", "有色金属I", "煤炭I", "钢铁I", "采掘I"}
        if not industries.intersection(I) and not self.is_empty_month(context):
            print("开仓")
            L = self.filter(context)
        else:
            print("跑")
            L = [self.fill_stock]
        log.info(self.name, '的选股列表:', L)
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
        log.info(self.name, '的选股列表:', targets)
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
                np.sum(weights * residuals ** 2) / np.sum(weights * (y - np.mean(y)) ** 2)
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
        select_stock = target[: min(len(target), self.stock_sum)]

        log.info(self.name, '的选股列表:', select_stock)
        return select_stock

    # 调仓
    def adjust(self, context):
        # 获取动量最高的一只ETF
        target = self.select()
        self._prepare(context)
        self._adjust(context, target)
