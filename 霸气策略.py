# 克隆自聚宽文章：https://www.joinquant.com/post/36235
# 标题：振幅选股 升级2，小改动升级
# 作者：玉米哥

# 克隆自聚宽文章：https://www.joinquant.com/post/36235
# 标题：振幅选股 升级2，小改动升级
# 作者：玉米哥

# 导入函数库
import jqdata
import pandas as pd


# 初始化函数，设定要操作的股票、基准等等
def initialize(context):
    # 定义一个全局变量, 保存要操作的股票
    # 000001(股票:平安银行)
    g.security = '000001.XSHE'
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    log.set_level('order', 'error')
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5),
                   type='stock')
    ###########设置公共参数
    g.stock_list = []
    g.stock_num = 5

    run_daily(before_market_open, 'before_open')
    run_daily(buy, 'every_bar')
    run_daily(sell, 'every_bar')


def before_market_open(context):
    # 选股
    t_days = jqdata.get_trade_days(end_date=context.previous_date, count=1)

    # 条件1，选择沪深300的股票入池
    stock_list = get_index_stocks('399101.XSHE', date=t_days[-1])
    # 条件2.5日振幅均线排行，降序
    df = get_price(stock_list, end_date=t_days[-1], fields=['low', 'pre_close', 'high'], count=1, frequency='1d',
                   skip_paused=False, fq='pre')
    z = (df.high.iloc[-1] - df.low.iloc[-1]) / df.pre_close.iloc[-1]
    log.info(z[:10])
    #    z = pd.DataFrame(z)
    #    z.columns = ['zf']
    #    s = z.sort_index(by='zf',ascending=False)
    stock_list = list(z.sort_values(ascending=False).index)
    log.info(stock_list[:10])
    #    stock_list = list(s.index)

    g.stock_list = stock_list[3:5]
    log.info(t_days[-1].strftime("%Y%m%d"), "选股：\n", g.stock_list)


def buy(context):
    for stock in g.stock_list:
        holds = context.portfolio.positions
        cash = context.portfolio.available_cash
        if stock not in context.portfolio.positions.keys() and len(context.portfolio.positions) < 3:
            buy_cash = cash / (2 - len(context.portfolio.positions))
            order_value(stock, buy_cash)
            log.info("Buying %s" % (stock))


def sell(context):
    # 1，30分钟下破Ma10均线
    for stock in context.portfolio.positions.keys():
        if context.portfolio.positions[stock].closeable_amount > 0:
            df = get_price(stock, end_date=context.current_dt, frequency='1d', fields=['high', 'close', 'volume'],
                           count=10)
            ma5 = pd.Series.rolling(df.close, window=6).mean()
            mv5 = pd.Series.rolling(df.volume, window=6).mean()
            if df.close[-1] < ma5[-1]:
                order_target(stock, 0)
                log.info("跌破5均线Selling %s" % (stock))
                continue
            # 2，高位大量，量比前5量均线大2倍以上，且出现长上影线,回撤超过3%
            elif df.volume[-1] > mv5[-2] * 2 and (df.high[-1] - df.close[-1]) / df.close[-1] > 0.03:
                order_target(stock, 0)
                log.info("巨量回撤Selling %s" % (stock))
            continue


def zhenfu(context, stock):
    # 算法（最高价-min（最低价，昨天收盘））/昨天收盘,5日振幅均线
    df = get_price(stock, end_date=context.previous_date, count=6, fields=['high', 'low', 'close'], skip_paused=False,
                   fq='pre')
    zhenfu = []
    for i in range(3, 6):
        low = min(df.close[-(i + 1)], df.low[-i])
        zf = (df.high[-i] - low) / df.close[-(i + 1)] * 100
        zhenfu.append(zf)
    zf_ma = mean(zhenfu)

    return zf_ma