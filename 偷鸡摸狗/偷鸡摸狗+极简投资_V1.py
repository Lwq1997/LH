# 克隆自聚宽文章：https://www.joinquant.com/post/51819
# 标题：多策略组合(加入极简价投，近五年年化50%，回撤20%)
# 作者：悬棋

# 克隆自聚宽文章：https://www.joinquant.com/post/51786
# 标题：多策略组合2.0参数修改优化
# 作者：lsfdz

# 克隆自聚宽文章：https://www.joinquant.com/post/51758
# 标题：多策略组合2.0（高手续费，近五年年化30%，回撤7%）
# 作者：美吉姆优秀毕业代表（重写）

# 克隆自聚宽文章：https://www.joinquant.com/post/51758
# 标题：多策略组合2.0（高手续费，近五年年化30%，回撤7%）
# 作者：O_iX

# 导入函数库
from jqdata import *  # 导入聚宽平台的函数库
import datetime  # 导入日期时间模块
import math  # 导入数学计算模块
import numpy as np  # 导入科学计算库
import pandas as pd  # 导入数据处理库


# 初始化函数，设定基准等
def initialize(context):
    # 设定基准指数
    set_benchmark("515080.XSHG")  # 设置基准为515080.XSHG
    set_option("avoid_future_data", True)  # 避免未来数据
    set_option("use_real_price", True)  # 使用真实价格
    log.info("初始函数开始运行且全局只运行一次")  # 打印日志
    log.set_level("order", "error")  # 设置订单日志级别为error
    set_slippage(FixedSlippage(0.02), type="stock")  # 设置股票交易滑点
    set_slippage(FixedSlippage(0.002), type="fund")  # 设置基金交易滑点
    # 设置股票交易成本
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003, close_commission=0.0003,
                             close_today_commission=0, min_commission=5), type="stock")
    # 设置货币基金交易成本
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0, close_commission=0, close_today_commission=0,
                             min_commission=0), type="mmf")

    # 初始化策略和子账户
    g.strategys = {}  # 存储策略实例的字典
    g.portfolio_value_proportion = [0, 0.4, 0.1, 0.1,
                                    0.4]  # 四个子账户资金顺序比例（保留现金：0%、搅屎棍策略：30%、全天候ETF策略：50%、核心资产轮动ETF策略：20%顺序不变数据可改）

    # 创建子账户
    set_subportfolios(
        [SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[i], "stock") for i in
         range(5)])

    # 初始化策略实例
    g.strategys["搅屎棍策略"] = JSG_Strategy(context, subportfolio_index=1, name="搅屎棍策略")  # 搅屎棍策略
    g.strategys["全天候策略"] = All_Day_Strategy(context, subportfolio_index=2, name="全天候策略")  # 全天候策略
    g.strategys["核心资产轮动策略"] = Rotation_ETF_Strategy(context, subportfolio_index=3,
                                                            name="核心资产轮动策略")  # 核心资产轮动策略
    g.strategys["破净策略"] = PJ_Strategy(context, subportfolio_index=4, name="破净策略")  # 破净策略

    # 定时运行函数
    run_monthly(balance_subportfolios, 1, "9:00")  # 每月1号9:00平衡子账户仓位
    if g.portfolio_value_proportion[1] > 0:  # 如果搅屎棍策略分配了资金
        run_daily(prepare_jsg_strategy, "9:05")  # 每天9:05准备搅屎棍策略
        run_weekly(adjust_jsg_strategy, 1, "9:31")  # 每周一9:31调整搅屎棍策略
        run_daily(check_jsg_strategy, "14:50")  # 每天14:50检查搅屎棍策略
    if g.portfolio_value_proportion[2] > 0:  # 如果全天候策略分配了资金
        run_monthly(adjust_all_day_strategy, 1, "9:40")  # 每月1号9:40调整全天候策略
    if g.portfolio_value_proportion[3] > 0:  # 如果核心资产轮动策略分配了资金
        run_daily(adjust_rotation_etf_strategy, "9:32")  # 每天9:32调整核心资产轮动策略
    if g.portfolio_value_proportion[4] > 0:  # 如果核心资产轮动策略分配了资金
        run_daily(prepare_pj_strategy, "9:05")  # 每天9:05准备破净策略
        run_daily(adjust_pj_strategy, "9:30")  # 每天9:30调整破净策略
        run_daily(check_pj_strategy, "14:00")  # 每天14:00检查破净策略
        run_daily(check_pj_strategy, "14:50")  # 每天14:50检查破净策略


# 搅屎棍策略相关函数
def prepare_jsg_strategy(context):
    # log.info("开始准备搅屎棍策略")  # 打印日志
    g.strategys["搅屎棍策略"].prepare(context)  # 调用搅屎棍策略的prepare方法


def adjust_jsg_strategy(context):
    # log.info("开始调整搅屎棍策略")  # 打印日志
    g.strategys["搅屎棍策略"].adjust(context)  # 调用搅屎棍策略的adjust方法


def check_jsg_strategy(context):
    # log.info("开始检查搅屎棍策略")  # 打印日志
    g.strategys["搅屎棍策略"].check(context)  # 调用搅屎棍策略的check方法


# 全天候策略相关函数
def adjust_all_day_strategy(context):
    # log.info("开始调整全天候策略")  # 打印日志
    g.strategys["全天候策略"].adjust(context)  # 调用全天候策略的adjust方法


# 核心资产轮动策略相关函数
def adjust_rotation_etf_strategy(context):
    # log.info("开始调整核心资产轮动策略")  # 打印日志
    g.strategys["核心资产轮动策略"].adjust(context)  # 调用核心资产轮动策略的adjust方法


# 破净策略相关函数
def prepare_pj_strategy(context):
    # log.info("开始准备破净策略")  # 打印日志
    g.strategys["破净策略"].prepare(context)  # 调用破净策略的prepare方法


def adjust_pj_strategy(context):
    # log.info("开始调整破净策略")  # 打印日志
    g.strategys["破净策略"].adjust(context)  # 调用破净策略的adjust方法


def check_pj_strategy(context):
    # log.info("开始破净策略")  # 打印日志
    g.strategys["破净策略"].check(context)  # 调用破净策略的check方法


# 平衡子账户仓位函数
def balance_subportfolios(context):
    log.info("开始平衡子账户仓位")  # 打印日志
    length = len(g.portfolio_value_proportion)  # 获取子账户数量
    log.info("调整前：" + str(
        [context.subportfolios[i].total_value / context.portfolio.total_value for i in range(length)]))  # 打印调整前的仓位比例
    for i in range(1, length):  # 遍历子账户
        target = g.portfolio_value_proportion[i] * context.portfolio.total_value  # 计算目标资金
        value = context.subportfolios[i].total_value  # 获取当前资金
        if context.subportfolios[i].available_cash > 0 and target < value:  # 如果当前资金超过目标资金
            transfer_cash(from_pindex=i, to_pindex=0,
                          cash=min(value - target, context.subportfolios[i].available_cash))  # 转移多余资金到主账户
    for i in range(1, length):  # 遍历子账户
        target = g.portfolio_value_proportion[i] * context.portfolio.total_value  # 计算目标资金
        value = context.subportfolios[i].total_value  # 获取当前资金
        if target > value and context.subportfolios[0].available_cash > 0:  # 如果当前资金不足
            transfer_cash(from_pindex=0, to_pindex=i,
                          cash=min(target - value, context.subportfolios[0].available_cash))  # 从主账户转移资金到子账户
    log.info("调整后：" + str(
        [context.subportfolios[i].total_value / context.portfolio.total_value for i in range(length)]))  # 打印调整后的仓位比例


# 策略基类
class Strategy:
    def __init__(self, context, subportfolio_index, name):
        self.subportfolio_index = subportfolio_index  # 子账户索引
        self.name = name  # 策略名称
        self.stock_sum = 1  # 最大持股数量
        self.hold_list = []  # 持仓列表
        self.limit_up_list = []  # 涨停股票列表
        self.fill_stock = "511880.XSHG"  # 默认填充股票（货币基金）
        self.portfolio_value = pd.DataFrame(columns=['date', 'total_value'])  # 记录每日总资产
        self.starting_cash = None  # 初始资金

    # 记录每日总资产
    def record_daily_value(self, context):
        subportfolio = context.subportfolios[self.subportfolio_index]  # 获取子账户
        new_data = {'date': context.current_dt.date(), 'total_value': subportfolio.total_value}  # 记录日期和总资产
        self.portfolio_value = self.portfolio_value.append(new_data, ignore_index=True)  # 添加到DataFrame中

        # 计算收益率
        if self.starting_cash is None:  # 如果是第一次运行，记录初始资金
            self.starting_cash = subportfolio.total_value
        ret_ratio = (subportfolio.total_value / self.starting_cash - 1) * 100  # 计算收益率

        # 记录分策略收益率
        title = self.name + '收益率'  # 设置曲线名称
        record(**{title: ret_ratio})  # 绘制折线图

    # 准备策略，获取持仓列表和涨停列表
    def _prepare(self, context):
        self.hold_list = list(context.subportfolios[self.subportfolio_index].long_positions.keys())  # 获取持仓股票列表
        if self.hold_list:  # 如果有持仓
            df = get_price(self.hold_list, end_date=context.previous_date, frequency="daily",
                           fields=["close", "high_limit"], count=1, panel=False, fill_paused=False)  # 获取股票价格
            df = df[df["close"] == df["high_limit"]]  # 筛选涨停股票
            self.limit_up_list = list(df.code)  # 记录涨停股票
        else:
            self.limit_up_list = []  # 如果没有持仓，清空涨停列表

    # 检查持仓股票是否涨停打开
    def _check(self, context):
        if self.limit_up_list:  # 如果有涨停股票
            current_data = get_current_data()  # 获取当前数据
            for stock in self.limit_up_list:  # 遍历涨停股票
                if current_data[stock].last_price < current_data[stock].high_limit:  # 如果涨停打开
                    log.info("[%s]涨停打开，卖出" % (stock))  # 打印日志
                    self.close_position(stock)  # 卖出股票
                else:
                    log.info("[%s]涨停，继续持有" % (stock))  # 打印日志

    # 调整持仓，买入目标股票，卖出不在目标列表中的股票
    def _adjust(self, context, target):
        subportfolio = context.subportfolios[self.subportfolio_index]  # 获取子账户
        for security in self.hold_list:  # 遍历持仓股票
            if (security not in target) and (security not in self.limit_up_list):  # 如果股票不在目标列表中且未涨停
                self.close_position(security)  # 卖出股票
        position_count = len(subportfolio.long_positions)  # 获取当前持仓数量
        if len(target) > position_count:  # 如果目标股票数量大于当前持仓数量
            buy_num = min(len(target), self.stock_sum - position_count)  # 计算可买入数量
            value = subportfolio.available_cash / buy_num  # 计算每只股票的买入金额
            for security in target:  # 遍历目标股票
                if security not in list(subportfolio.long_positions.keys()):  # 如果股票未持仓
                    if self.open_position(security, value):  # 买入股票
                        if position_count == len(target):  # 如果达到最大持仓数量
                            break

    # 基础股票池
    def stockpool_index(self, context, pool_id=1):
        lists = list(get_all_securities(types=['stock'], date=context.previous_date).index)
        if pool_id == 0:
            pass
        elif pool_id == 1:
            current_data = get_current_data()
            lists = [stock for stock in lists if not (
                    (current_data[stock].day_open == current_data[stock].high_limit) or  # 涨停开盘
                    (current_data[stock].day_open == current_data[stock].low_limit) or  # 跌停开盘
                    current_data[stock].paused or  # 停牌
                    current_data[stock].is_st or  # ST
                    ('ST' in current_data[stock].name) or
                    ('*' in current_data[stock].name) or
                    ('退' in current_data[stock].name) or
                    (stock.startswith('30')) or  # 创业
                    (stock.startswith('68')) or  # 科创
                    (stock.startswith('8')) or  # 北交
                    (stock.startswith('4'))  # 北交
            )
                     ]

        return lists

    # 下单，调整股票仓位到目标价值
    def order_target_value_(self, security, value):
        if value == 0:  # 如果目标价值为0
            log.debug("Selling out %s" % (security))  # 打印日志
        else:
            log.debug("Order %s to value %f" % (security, value))  # 打印日志
        return order_target_value(security, value, pindex=self.subportfolio_index)  # 下单

    # 开仓
    def open_position(self, security, value):
        order = self.order_target_value_(security, value)  # 下单
        if order is not None and order.filled > 0:  # 如果订单成交
            return True
        return False

    # 平仓
    def close_position(self, security):
        order = self.order_target_value_(security, 0)  # 下单
        if order is not None:  # 如果订单存在
            if order.status == OrderStatus.held and order.filled == order.amount:  # 如果订单完全成交
                return True
        return False

    # 过滤停牌股票
    def filter_paused_stock(self, stock_list):
        current_data = get_current_data()  # 获取当前数据
        return [stock for stock in stock_list if not current_data[stock].paused]  # 过滤停牌股票

    # 过滤ST股票
    def filter_st_stock(self, stock_list):
        current_data = get_current_data()  # 获取当前数据
        return [stock for stock in stock_list if
                not current_data[stock].is_st and "ST" not in current_data[stock].name and "*" not in current_data[
                    stock].name and "退" not in current_data[stock].name]  # 过滤ST股票

    # 过滤科创板和北交所股票
    def filter_kcbj_stock(self, stock_list):
        for stock in stock_list[:]:  # 遍历股票列表
            if stock[0] == "4" or stock[0] == "8" or stock[:2] == "68" or stock[0] == "3":  # 过滤科创板和北交所股票
                stock_list.remove(stock)
        return stock_list

    # 过滤涨停和跌停股票
    def filter_limitup_limitdown_stock(self, context, stock_list):
        subportfolio = context.subportfolios[self.subportfolio_index]  # 获取子账户
        current_data = get_current_data()  # 获取当前数据
        return [stock for stock in stock_list if stock in subportfolio.long_positions.keys() or (
                    current_data[stock].last_price < current_data[stock].high_limit and current_data[stock].last_price >
                    current_data[stock].low_limit)]  # 过滤涨跌停股票

    # 过滤新股
    def filter_new_stock(self, context, stock_list, days):
        yesterday = context.previous_date  # 获取前一天日期
        filtered_stocks = [stock for stock in stock_list if
                           not yesterday - get_security_info(stock).start_date < datetime.timedelta(days)]  # 过滤新股
        # log.info(f"过滤新股后剩余股票数量：{len(filtered_stocks)}")  # 打印日志
        return filtered_stocks

    # 过滤高价股
    def filter_high_price_stock(self, stock_list):
        last_prices = history(1, unit="1m", field="close", security_list=stock_list).iloc[0]  # 获取最新价格
        filtered_stocks = last_prices[last_prices < 10].index.tolist()  # 过滤高价股
        # log.info(f"过滤高价股后剩余股票数量：{len(filtered_stocks)}")  # 打印日志
        return filtered_stocks


# 搅屎棍策略
class JSG_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)  # 调用父类构造函数
        self.stock_sum = 10  # 最大持股数量（数量可变）
        self.num = 1  # 行业数量
        self.pass_months = [1, 4]  # 空仓月份

    # 获取股票行业
    def getStockIndustry(self, stocks):
        industry = get_industry(stocks)  # 获取股票行业信息
        dict = {
            stock: info["sw_l1"]["industry_name"]
            for stock, info in industry.items()
            if "sw_l1" in info
        }
        return pd.Series(dict)  # 返回行业信息

    # 计算全市场宽度
    def get_market_breadth(self, context):
        yesterday = context.previous_date  # 获取前一天日期
        stocks = get_index_stocks("000985.XSHG")  # 获取指数成分股
        count = 1  # 计算周期
        h = get_price(
            stocks,
            end_date=yesterday,
            frequency="1d",
            fields=["close"],
            count=count + 20,
            panel=False,
        )  # 获取股票价格
        h["date"] = pd.DatetimeIndex(h.time).date  # 转换日期格式
        df_close = h.pivot(index="code", columns="date", values="close").dropna(axis=0)  # 转换数据格式
        df_ma20 = df_close.rolling(window=20, axis=1).mean().iloc[:, -count:]  # 计算20日均线
        df_bias = df_close.iloc[:, -count:] > df_ma20  # 计算是否高于均线
        df_bias["industry_name"] = self.getStockIndustry(stocks)  # 添加行业信息
        df_ratio = (
                (df_bias.groupby("industry_name").sum() * 100.0)
                / df_bias.groupby("industry_name").count()
        ).round()  # 计算行业热度
        top_values = df_ratio.loc[:, yesterday].nlargest(self.num)  # 获取热度最高的行业
        I = top_values.index.tolist()  # 转换为列表
        # log.info(f"全市场宽度：{np.array(df_ratio.sum(axis=0).mean())}")  # 打印全市场宽度
        # log.info(f"行业热度排名前{self.num}的行业：{[name for name in I]}")  # 打印行业热度
        return I

    # 过滤股票
    def filter(self, context):
        stocks = get_index_stocks("399101.XSHE", context.current_dt)  # 获取指数成分股
        # log.info(f"初始股票池数量：{len(stocks)}")  # 打印日志
        stocks = self.filter_kcbj_stock(stocks)  # 过滤科创板和北交所股票
        # log.info(f"过滤科创板股票后剩余股票数量：{len(stocks)}")  # 打印日志
        stocks = self.filter_st_stock(stocks)  # 过滤ST股票
        # log.info(f"过滤ST股票后剩余股票数量：{len(stocks)}")  # 打印日志
        stocks = self.filter_new_stock(context, stocks, 375)  # 过滤新股
        stocks = self.filter_paused_stock(stocks)  # 过滤停牌股票
        # log.info(f"过滤停牌股票后剩余股票数量：{len(stocks)}")  # 打印日志
        stocks = get_fundamentals(
            query(
                valuation.code,
            )
            .filter(
                valuation.code.in_(stocks),
                income.np_parent_company_owners > 0,
                income.net_profit > 0,
                income.operating_revenue > 1e8,
            )
            .order_by(valuation.market_cap.asc())
        )["code"].tolist()  # 根据财务指标过滤股票
        # log.info(f"根据财务指标过滤后剩余股票数量：{len(stocks)}")  # 打印日志
        stocks = self.filter_limitup_limitdown_stock(context, stocks)  # 过滤涨跌停股票
        # log.info(f"过滤涨跌停股票后剩余股票数量：{len(stocks)}")  # 打印日志
        selected_stocks = stocks[: min(len(stocks), self.stock_sum)]  # 选择最终股票
        # log.info(f"最终选择的股票数量：{len(selected_stocks)}")  # 打印日志
        return selected_stocks

    # 判断是否为空仓月份
    def is_empty_month(self, context):
        month = context.current_dt.month  # 获取当前月份
        return month in self.pass_months  # 判断是否为空仓月份

    # 选股
    def select(self, context):
        I = self.get_market_breadth(context)  # 获取行业热度
        industries = {"银行I", "有色金属I", "煤炭I", "钢铁I", "采掘I"}  # 定义特定行业
        if not industries.intersection(I) and not self.is_empty_month(context):  # 如果不在特定行业且不是空仓月份
            # log.info("开仓")  # 打印日志
            L = self.filter(context)  # 过滤股票
        else:
            # log.info("跑")  # 打印日志
            L = [self.fill_stock]  # 使用默认填充股票
        return L

    def prepare(self, context):
        self._prepare(context)  # 调用父类的prepare方法

    def adjust(self, context):
        target = self.select(context)  # 选股
        self._adjust(context, target)  # 调整持仓
        self.record_daily_value(context)  # 记录每日总资产

    def check(self, context):
        self._check(context)  # 调用父类的check方法


# 全天候策略
class All_Day_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)  # 调用父类构造函数
        self.min_volume = 2000  # 最小交易量
        self.etf_pool = [
            "511010.XSHG",  # 国债ETF
            "518880.XSHG",  # 黄金ETF
            "513100.XSHG",  # 纳指ETF
            "515080.XSHG",  # 创新药ETF
            "159980.XSHE",  # 有色金属ETF
            "162411.XSHE",  # 华宝油气
            "159985.XSHE",  # 豆粕ETF
        ]
        self.rates = [0.4, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05]  # ETF权重

    def adjust(self, context):
        subportfolio = context.subportfolios[self.subportfolio_index]  # 获取子账户
        targets = {
            etf: subportfolio.total_value * rate
            for etf, rate in zip(self.etf_pool, self.rates)
        }  # 计算目标持仓价值
        if not subportfolio.long_positions:  # 如果没有持仓
            # log.info("无持仓，按目标权重下单")  # 打印日志
            for etf, target in targets.items():  # 遍历ETF
                self.order_target_value_(etf, target)  # 下单
        else:
            # log.info("有持仓，调整仓位")  # 打印日志
            for etf, target in targets.items():  # 遍历ETF
                value = subportfolio.long_positions[etf].value  # 获取当前持仓价值
                minV = subportfolio.long_positions[etf].price * 100  # 计算最小交易量
                if value - target > self.min_volume and minV > value - target:  # 如果持仓超过目标
                    log.info(f"减少{etf}的仓位")  # 打印日志
                    self.order_target_value_(etf, target)  # 调整仓位
            for etf, target in targets.items():  # 遍历ETF
                value = subportfolio.long_positions[etf].value  # 获取当前持仓价值
                minV = subportfolio.long_positions[etf].price * 100  # 计算最小交易量
                if (
                        target - value > self.min_volume
                        and minV < subportfolio.available_cash
                        and minV < target - value
                ):  # 如果持仓不足
                    log.info(f"增加{etf}的仓位")  # 打印日志
                    self.order_target_value_(etf, target)  # 调整仓位
        self.record_daily_value(context)  # 记录每日总资产


# 核心资产轮动策略
class Rotation_ETF_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)  # 调用父类构造函数
        self.stock_sum = 1  # 最大持股数量
        self.etf_pool = [
            "518880.XSHG",  # 黄金ETF
            "513100.XSHG",  # 纳指ETF
            "159915.XSHE",  # 创业板ETF
            "510180.XSHG",  # 上证180ETF
        ]
        self.m_days = 25  # 动量计算周期

    # 计算动量
    def MOM(self, etf):
        df = attribute_history(etf, self.m_days, "1d", ["close"])  # 获取历史价格
        y = np.log(df["close"].values)  # 对数收益率
        n = len(y)  # 数据长度
        x = np.arange(n)  # 时间序列
        weights = np.linspace(1, 2, n)  # 权重
        slope, intercept = np.polyfit(x, y, 1, w=weights)  # 线性拟合
        annualized_returns = math.pow(math.exp(slope), 250) - 1  # 计算年化收益率
        residuals = y - (slope * x + intercept)  # 残差
        r_squared = 1 - (
                np.sum(weights * residuals ** 2) / np.sum(weights * (y - np.mean(y)) ** 2)
        )  # 计算R平方
        return annualized_returns * r_squared  # 返回动量值

    # 选股
    def select(self):
        score_list = [self.MOM(etf) for etf in self.etf_pool]  # 计算动量值
        df = pd.DataFrame(index=self.etf_pool, data={"score": score_list})  # 创建DataFrame
        df = df.sort_values(by="score", ascending=False)  # 按动量值排序
        df = df[(df["score"] > 0) & (df["score"] <= 5)]  # 过滤动量值
        target = df.index.tolist()  # 获取目标ETF
        log.info(f"根据动量策略选择的ETF：{target}")  # 打印日志
        if not target:  # 如果没有目标ETF
            target = [self.fill_stock]  # 使用默认填充股票
        return target[: min(len(target), self.stock_sum)]  # 返回目标ETF

    def adjust(self, context):
        target = self.select()  # 选股
        self._prepare(context)  # 调用父类的prepare方法
        self._adjust(context, target)  # 调整持仓
        self.record_daily_value(context)  # 记录每日总资产


# # PB策略
# class PJ_Strategy(Strategy):
#     def __init__(self, context, subportfolio_index, name):
#         super().__init__(context, subportfolio_index, name)  # 调用父类构造函数
#         self.stock_sum = 1

#     def select(self, context):
#         self.select_list = self.__get_rank(context)[:self.stock_sum]
#         #self.print_trade_plan(context, self.select_list)                                回测时取消打印
#         def __get_rank(self, context):
#         lists = self.stockpool_index(context)
#         # 基本股选股
#         q = query(
#                 valuation.code, valuation.market_cap, valuation.pe_ratio, income.total_operating_revenue
#             ).filter(
#                 valuation.pb_ratio < 1,
#                 cash_flow.subtotal_operate_cash_inflow > 1e6,
#                 indicator.adjusted_profit > 1e6,
#                 indicator.roa > 0.15,
#                 indicator.inc_operation_profit_year_on_year > 0,
#             	valuation.code.in_(lists)
#         	).order_by(
#         	    indicator.roa.desc()
#             ).limit(
#             	self.stock_sum * 3
#             )
#         lists = list(get_fundamentals(q).code)
#         return lists

#     def prepare(self, context):
#         self._prepare(context)  # 调用父类的prepare方法

#     def adjust(self, context):
#         target = self.select(context)  # 选股
#         self._adjust(context, target)  # 调整持仓
#         self.record_daily_value(context)  # 记录每日总资产

#     def check(self, context):
#         self._check(context)  # 调用父类的check方法
# PB策略
class PJ_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)  # 调用父类构造函数
        self.stock_sum = 1  # 最大持股数量

    def select(self, context):
        # 调用选股逻辑
        self.select_list = self.__get_rank(context)[:self.stock_sum]
        log.info(self.name, '的选股列表:', self.select_list)
        return self.select_list  # 返回选股列表

    def __get_rank(self, context):
        # 获取股票池
        lists = self.stockpool_index(context)
        q = query(
            valuation.code, valuation.market_cap, valuation.pe_ratio, income.total_operating_revenue
        ).filter(
            valuation.pb_ratio < 1,  # 破净
            cash_flow.subtotal_operate_cash_inflow > 1e6,  # 经营现金流
            indicator.adjusted_profit > 1e6,  # 扣非净利润
            indicator.roa > 0.15,  # 总资产收益率
            indicator.inc_operation_profit_year_on_year > 0,  # 净利润同比增长
            valuation.code.in_(lists)
        ).order_by(
            indicator.roa.desc()  # 按ROA降序排序
        ).limit(
            self.stock_sum * 3  # 限制选股数量
        )
        lists = list(get_fundamentals(q).code)  # 获取选股列表
        return lists

    def prepare(self, context):
        self._prepare(context)  # 调用父类的prepare方法

    def adjust(self, context):
        target = self.select(context)  # 选股
        if target is None or len(target) == 0:  # 如果选股列表为空
            target = '511010.XSHG'  # 使用默认填充股票
        self._adjust(context, target)  # 调整持仓
        self.record_daily_value(context)  # 记录每日总资产

    def check(self, context):
        self._check(context)  # 调用父类的check方法