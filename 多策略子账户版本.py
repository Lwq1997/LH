# 克隆自聚宽文章：https://www.joinquant.com/post/50948
# 标题：（吼-多策略子账户工具）此消彼长-多策略子账户实现
# 作者：kautz

# 克隆自聚宽文章：https://www.joinquant.com/post/47330
# 标题：此消彼长
# 作者：明曦

# 克隆自聚宽文章：https://www.joinquant.com/post/47344
# 标题：用子账户模拟多策略分仓
# 作者：赌神Buffett

'''
多策略分子账户并行

用到的策略：
DSZMX_strategy：明曦大市值策略
XSZMX_strategy：明曦小市值策略

'''
# 导入函数库
from jqdata import *
from jqfactor import get_factor_values
import datetime


# 初始化函数，设定基准等等
def initialize(context):
    log.warn('--initialize函数(只运行一次)--',
             str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 过滤掉order系列API产生的比error级别低的log
    log.set_level('order', 'error')
    # 关闭未来函数
    set_option('avoid_future_data', True)

    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0001, close_commission=0.0001, min_commission=0),
                   type='stock')

    # 为股票设定滑点为百分比滑点
    set_slippage(PriceRelatedSlippage(0.01), type='stock')

    # 临时变量

    # 持久变量
    g.strategys = {}
    g.portfolio_value_proportion = [0.7, 0.3]
    g.portfolio_value_proportion = [0.5, 0.5]
    g.portfolio_value_proportion = [0.3, 0.7]

    # 创建策略实例
    set_subportfolios([
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[0], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[1], 'stock'),
    ])

    params = {
        'max_hold_count': 5,  # 最大持股数
        'max_select_count': 5,  # 最大输出选股数
    }
    xszMX_strategy = XSZMX_strategy(context, subportfolio_index=0, name='明曦小市值策略', params=params)
    g.strategys[xszMX_strategy.name] = xszMX_strategy

    params = {
        'max_hold_count': 5,  # 最大持股数
        'max_select_count': 5,  # 最大输出选股数
    }
    dszMX_strategy = DSZMX_strategy(context, subportfolio_index=1, name='明曦大市值策略', params=params)
    g.strategys[dszMX_strategy.name] = dszMX_strategy

    # 执行计划
    if g.portfolio_value_proportion[0] > 0:
        # run_daily(XSZMX_prepare, time='7:30')
        # run_daily(XSZMX_select, time='7:40')
        run_monthly(XSZMX_select, 1, time='7:40')
        # run_daily(XSZMX_open_market, time='9:30')
        # run_daily(XSZMX_adjust, time='10:00')
        run_monthly(XSZMX_adjust, 1, time='9:35')
        # run_daily(XSZMX_sell_when_highlimit_open, time='14:00')
        # run_daily(XSZMX_sell_when_highlimit_open, time='14:50')
    if g.portfolio_value_proportion[1] > 0:
        # run_daily(DSZMX_prepare, time='7:30')
        run_monthly(DSZMX_select, 1, time='7:40')
        # run_daily(DSZMX_open_market, time='9:30')
        run_monthly(DSZMX_adjust, 1, time='9:35')
        # run_daily(DSZMX_sell_when_highlimit_open, time='14:00')
        # run_daily(DSZMX_sell_when_highlimit_open, time='14:50')

    # run_daily(print_trade_info, time='15:01')


def XSZMX_prepare(context):
    g.strategys['明曦小市值策略'].day_prepare(context)


def XSZMX_select(context):
    g.strategys['明曦小市值策略'].select(context)


def XSZMX_adjust(context):
    g.strategys['明曦小市值策略'].adjustwithnoRM(context)


def XSZMX_open_market(context):
    g.strategys['明曦小市值策略'].close_for_stoplost(context)


def XSZMX_sell_when_highlimit_open(context):
    g.strategys['明曦小市值策略'].sell_when_highlimit_open(context)


def DSZMX_prepare(context):
    g.strategys['明曦大市值策略'].day_prepare(context)


def DSZMX_select(context):
    g.strategys['明曦大市值策略'].select(context)


def DSZMX_adjust(context):
    g.strategys['明曦大市值策略'].adjustwithnoRM(context)


def DSZMX_open_market(context):
    g.strategys['明曦大市值策略'].close_for_stoplost(context)


def DSZMX_sell_when_highlimit_open(context):
    g.strategys['明曦大市值策略'].sell_when_highlimit_open(context)


# 打印交易记录
def print_trade_info(context):
    orders = get_orders()
    for _order in orders.values():
        print('成交记录：' + str(_order))


# 策略基类
# 同一只股票只买入1次，卖出时全部卖出
class Strategy:
    def __init__(self, context, subportfolio_index, name, params):
        self.subportfolio_index = subportfolio_index
        # self.subportfolio = context.subportfolios[subportfolio_index]
        self.name = name
        self.params = params
        self.max_hold_count = self.params['max_hold_count'] if 'max_hold_count' in self.params else 1  # 最大持股数
        self.max_select_count = self.params['max_select_count'] if 'max_select_count' in self.params else 5  # 最大输出选股数
        self.hold_limit_days = self.params['hold_limit_days'] if 'hold_limit_days' in self.params else 20  # 计算最近持有列表的天数
        self.use_empty_month = self.params['use_empty_month'] if 'use_empty_month' in self.params else False  # 是否有空仓期
        self.empty_month = self.params['empty_month'] if 'empty_month' in self.params else []  # 空仓月份
        self.use_stoplost = self.params['use_stoplost'] if 'use_stoplost' in self.params else False  # 是否使用止损
        self.stoplost_silent_days = self.params[
            'stoplost_silent_days'] if 'stoplost_silent_days' in self.params else 20  # 止损后不交易的天数
        self.stoplost_level = self.params['stoplost_level'] if 'stoplost_level' in self.params else 0.2  # 止损的下跌幅度（按买入价）

        self.select_list = []
        self.hold_list = []  # 昨收持仓
        self.history_hold_list = []  # 最近持有列表
        self.not_buy_again_list = []  # 最近持有不再购买列表
        self.yestoday_high_limit_list = []  # 昨日涨停列表
        self.stoplost_date = None  # 止损日期，为None是表示未进入止损

    def day_prepare(self, context):
        subportfolio = context.subportfolios[self.subportfolio_index]

        # 获取昨日持股列表
        self.hold_list = list(subportfolio.long_positions)

        # 获取最近一段时间持有过的股票列表
        self.history_hold_list.append(self.hold_list)
        if len(self.history_hold_list) >= self.hold_limit_days:
            self.history_hold_list = self.history_hold_list[-self.hold_limit_days:]
        temp_set = set()
        for lists in self.history_hold_list:
            for stock in lists:
                temp_set.add(stock)
        self.not_buy_again_list = list(temp_set)

        # 获取昨日持股涨停列表
        if self.hold_list != []:
            df = get_price(self.hold_list, end_date=context.previous_date, frequency='daily',
                           fields=['close', 'high_limit'], count=1, panel=False, fill_paused=False)
            df = df[df['close'] == df['high_limit']]
            self.yestoday_high_limit_list = list(df.code)
        else:
            self.yestoday_high_limit_list = []

        # 检查空仓期
        self.check_empty_month(context)
        # 检查止损
        self.check_stoplost(context)

    # 基础股票池
    def stockpool(self, context, pool_id=1):
        lists = list(get_all_securities(types=['stock'], date=context.previous_date).index)
        if pool_id == 0:
            pass
        elif pool_id == 1:
            lists = self.filter_kcbj_stock(lists)
            lists = self.filter_st_stock(lists)
            lists = self.filter_paused_stock(lists)
            lists = self.filter_highlimit_stock(context, lists)
            lists = self.filter_lowlimit_stock(context, lists)

        return lists

    # 选股
    def select(self, context):
        # 空仓期控制
        if self.use_empty_month and context.current_dt.month in (self.empty_month):
            return
        # 止损期控制
        if self.stoplost_date is not None:
            return
        select.select_list = []

    # 打印交易计划
    def print_trade_plan(self, context, select_list):
        subportfolio = context.subportfolios[self.subportfolio_index]
        current_data = get_current_data()  # 取股票名称

        content = context.current_dt.date().strftime("%Y-%m-%d") + ' ' + self.name + " 交易计划：" + "\n"

        for stock in subportfolio.long_positions:
            if stock not in select_list[:self.max_hold_count]:
                content = content + stock + ' ' + current_data[stock].name + ' 卖出\n'

        for stock in select_list:
            if stock not in subportfolio.long_positions and stock in select_list[:self.max_hold_count]:
                content = content + stock + ' ' + current_data[stock].name + ' 买入\n'
            elif stock in subportfolio.long_positions and stock in select_list[:self.max_hold_count]:
                content = content + stock + ' ' + current_data[stock].name + ' 继续持有\n'
            else:
                content = content + stock + ' ' + current_data[stock].name + '\n'

        if ('买' in content) or ('卖' in content):
            print(content)

    # 调仓
    def adjust(self, context):
        # 空仓期控制
        if self.use_empty_month and context.current_dt.month in (self.empty_month):
            return
        # 止损期控制
        if self.stoplost_date is not None:
            return

        # 先卖后买
        hold_list = list(context.subportfolios[self.subportfolio_index].long_positions)
        sell_stocks = []
        for stock in hold_list:
            if stock not in self.select_list[:self.max_hold_count]:
                sell_stocks.append(stock)
        self.sell(context, sell_stocks)
        self.buy(context, self.select_list)

    # 涨停打开卖出
    def sell_when_highlimit_open(self, context):
        if self.yestoday_high_limit_list != []:
            for stock in self.yestoday_high_limit_list:
                if stock in context.subportfolios[self.subportfolio_index].long_positions:
                    current_data = get_price(stock, end_date=context.current_dt, frequency='1m',
                                             fields=['close', 'high_limit'],
                                             skip_paused=False, fq='pre', count=1, panel=False, fill_paused=True)
                    if current_data.iloc[0, 0] < current_data.iloc[0, 1]:
                        self.sell(context, [stock])
                        content = context.current_dt.date().strftime(
                            "%Y-%m-%d") + ' ' + self.name + ': {}涨停打开，卖出'.format(stock) + "\n"
                        print(content)

    # 空仓期检查
    def check_empty_month(self, context):
        subportfolio = context.subportfolios[self.subportfolio_index]
        if self.use_empty_month and context.current_dt.month in (self.empty_month) and len(
                subportfolio.long_positions) > 0:
            content = context.current_dt.date().strftime("%Y-%m-%d") + self.name + ': 进入空仓期' + "\n"
            for stock in subportfolio.long_positions:
                content = content + stock + "\n"
            print(content)

    # 进入空仓期清仓
    def close_for_empty_month(self, context):
        subportfolio = context.subportfolios[self.subportfolio_index]
        if self.use_empty_month and context.current_dt.month in (self.empty_month) and len(
                subportfolio.long_positions) > 0:
            self.sell(context, list(subportfolio.long_positions))

    # 止损检查
    def check_stoplost(self, context):
        subportfolio = context.subportfolios[self.subportfolio_index]
        if self.use_stoplost:
            if self.stoplost_date is None:
                last_prices = history(1, unit='1m', field='close', security_list=subportfolio.long_positions)
                for stock in subportfolio.long_positions:
                    position = subportfolio.long_positions[stock]
                    if (position.avg_cost - last_prices[stock][-1]) / position.avg_cost > self.stoplost_level:
                        self.stoplost_date = context.current_dt.date()
                        print(self.name + ': ' + '开始止损')
                        content = context.current_dt.date().strftime("%Y-%m-%d") + ' ' + self.name + ': 止损' + "\n"
                        for stock in subportfolio.long_positions:
                            content = content + stock + "\n"
                        print(content)
                        break
            else:  # 已经在清仓静默期
                if (context.current_dt + datetime.timedelta(
                        days=-self.stoplost_silent_days)).date() >= self.stoplost_date:
                    self.stoplost_date = None
                    print(self.name + ': ' + '退出止损')

    # 止损时清仓
    def close_for_stoplost(self, context):
        subportfolio = context.subportfolios[self.subportfolio_index]
        if self.use_stoplost and self.stoplost_date is not None and len(subportfolio.long_positions) > 0:
            self.sell(context, list(subportfolio.long_positions))

    # 买入多只股票
    def buy(self, context, buy_stocks):
        subportfolio = context.subportfolios[self.subportfolio_index]
        buy_count = self.max_hold_count - len(subportfolio.long_positions)
        if buy_count > 0:
            value = subportfolio.available_cash / buy_count
            index = 0
            for stock in buy_stocks:
                if stock in subportfolio.long_positions:
                    continue
                self.__open_position(stock, value)
                index = index + 1
                if index >= buy_count:
                    break

    # 卖出多只股票
    def sell(self, context, sell_stocks):
        subportfolio = context.subportfolios[self.subportfolio_index]
        for stock in sell_stocks:
            if stock in subportfolio.long_positions:
                self.__close_position(stock)

    # 开仓单只
    def __open_position(self, security, value):
        order = order_target_value(security, value, pindex=self.subportfolio_index)
        if order != None and order.filled > 0:
            return True
        return False

    # 清仓单只
    def __close_position(self, security):
        order = order_target_value(security, 0, pindex=self.subportfolio_index)
        if order != None and order.status == OrderStatus.held and order.filled == order.amount:
            return True
        return False

    # 过滤科创北交
    def filter_kcbj_stock(self, stock_list):
        for stock in stock_list[:]:
            if stock[0] == '4' or stock[0] == '8' or stock[:2] == '68':
                stock_list.remove(stock)
        return stock_list

    # 过滤停牌股票
    def filter_paused_stock(self, stock_list):
        current_data = get_current_data()
        return [stock for stock in stock_list if not current_data[stock].paused]

    # 过滤ST及其他具有退市标签的股票
    def filter_st_stock(self, stock_list):
        current_data = get_current_data()
        return [stock for stock in stock_list
                if not current_data[stock].is_st
                and 'ST' not in current_data[stock].name
                and '*' not in current_data[stock].name
                and '退' not in current_data[stock].name]

    # 过滤涨停的股票
    def filter_highlimit_stock(self, context, stock_list):
        subportfolio = context.subportfolios[self.subportfolio_index]
        last_prices = history(1, unit='1m', field='close', security_list=stock_list)
        current_data = get_current_data()

        # 已存在于持仓的股票即使涨停也不过滤，避免此股票再次可买，但因被过滤而导致选择别的股票
        return [stock for stock in stock_list if stock in subportfolio.long_positions
                or last_prices[stock][-1] < current_data[stock].high_limit]

    # 过滤跌停的股票
    def filter_lowlimit_stock(self, context, stock_list):
        subportfolio = context.subportfolios[self.subportfolio_index]
        last_prices = history(1, unit='1m', field='close', security_list=stock_list)
        current_data = get_current_data()

        return [stock for stock in stock_list if stock in subportfolio.long_positions
                or last_prices[stock][-1] > current_data[stock].low_limit]

    # 过滤次新股
    def filter_new_stock(self, context, stock_list, days):
        return [stock for stock in stock_list if
                not context.previous_date - get_security_info(stock).start_date < datetime.timedelta(days=days)]

    # 过滤大幅解禁
    def filter_locked_shares(self, context, stock_list, days):
        df = get_locked_shares(stock_list=stock_list, start_date=context.previous_date.strftime('%Y-%m-%d'),
                               forward_count=days)
        df = df[df['rate1'] > 0.2]  # 解禁数量占总股本的百分比
        filterlist = list(df['code'])
        return [stock for stock in stock_list if stock not in filterlist]


# DSZMX策略
class DSZMX_strategy(Strategy):
    def select(self, context):
        self.select_list = self.__get_rank(context)[:self.max_select_count]
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        lists = self.stockpool(context)

        # 基本股选股
        q = query(
            valuation.code, valuation.market_cap, valuation.pe_ratio, income.total_operating_revenue
        ).filter(
            valuation.code.in_(lists),
            valuation.pe_ratio_lyr.between(0, 30),  # 市盈率
            valuation.ps_ratio.between(0, 8),  # 市销率TTM
            valuation.pcf_ratio < 10,  # 市现率TTM
            indicator.eps > 0.3,  # 每股收益
            indicator.roe > 0.1,  # 净资产收益率
            indicator.net_profit_margin > 0.1,  # 销售净利率
            indicator.gross_profit_margin > 0.3,  # 销售毛利率
            indicator.inc_revenue_year_on_year > 0.25  # 营业收入同比增长率
        ).order_by(
            valuation.market_cap.desc()
        ).limit(
            self.max_select_count * 3
        )
        lists = list(get_fundamentals(q).code)
        return lists


# 小市值策略
class XSZMX_strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.new_days = 375  # 已上市天数
        self.factor_list = [
            (  ###
                [
                    'non_recurring_gain_loss',
                    'non_operating_net_profit_ttm',
                    'roe_ttm_8y',
                    'sharpe_ratio_20'
                ],
                [
                    -1.3651516084272432e-13,
                    -3.673549665003535e-14,
                    -0.006872269236387061,
                    -3.922028093095638e-12
                ]
            ),
        ]

    def select(self, context):
        # 空仓期控制
        if self.use_empty_month and context.current_dt.month in (self.empty_month):
            return
        # 止损期控制
        if self.stoplost_date is not None:
            return
        self.select_list = self.__get_rank(context)[:self.max_select_count]
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        initial_list = self.stockpool(context)
        initial_list = self.filter_new_stock(context, initial_list, self.new_days)
        # initial_list = self.filter_locked_shares(context, initial_list, 120)    # 过滤即将大幅解禁

        final_list = []
        # MS
        for factor_list, coef_list in self.factor_list:
            factor_values = get_factor_values(initial_list, factor_list, end_date=context.previous_date, count=1)
            df = pd.DataFrame(index=initial_list, columns=factor_values.keys())
            for i in range(len(factor_list)):
                df[factor_list[i]] = list(factor_values[factor_list[i]].T.iloc[:, 0])
            df = df.dropna()

            df['total_score'] = 0
            for i in range(len(factor_list)):
                df['total_score'] += coef_list[i] * df[factor_list[i]]
            df = df.sort_values(by=['total_score'], ascending=False)  # 分数越高即预测未来收益越高，排序默认降序
            complex_factor_list = list(df.index)[:int(0.1 * len(list(df.index)))]
            q = query(
                valuation.code, valuation.circulating_market_cap, indicator.eps
            ).filter(
                valuation.code.in_(complex_factor_list)
            )
            # .order_by(
            #     valuation.circulating_market_cap.asc()
            # )
            df = get_fundamentals(q)
            df = df[df['eps'] > 0]
            lst = list(df.code)
            final_list = list(set(final_list + lst))

        # 再做一次市值过滤
        q = query(valuation.code). \
            filter(valuation.code.in_(final_list)). \
            order_by(valuation.circulating_market_cap.asc())
        df = get_fundamentals(q)
        final_list = list(df.code)
        final_list = final_list[:min(self.max_select_count, len(final_list))]
        return final_list
