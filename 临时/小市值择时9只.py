# 克隆自聚宽文章：https://www.joinquant.com/post/53990
# 标题：二十年暴涨21000倍，惊呆了，无未来，正在实盘
# 作者：寒菱投资

# 克隆自聚宽文章：https://www.joinquant.com/post/53578
# 标题：关于小市值择时策略的研究之一
# 作者：Ceng-Lucifffff

# -*- coding: utf-8 -*-
"""
【老代码新写之稳健小市值择时】策略整理版
作者：Ceng-Lucifffff

本代码主要包含以下部分：
1. 数据辅助类（DataHelper）：封装了聚宽数据接口的调用（get_price 与 history），对异常进行捕获，输出中文日志。
2. 交易策略类（TradingStrategy）：封装选股、调仓、买卖、止损、风控等核心逻辑。利用属性管理持仓、候选股票及当前状态。
3. 全局包装函数：为调度任务提供顶层包装函数，确保调度时能通过序列化。
4. 初始化函数（initialize）：设置策略运行环境，并注册各个调度任务。

初学者可参考注释慢慢理解每个步骤的含义及执行流程。
"""

from typing import Any, List, Dict, Optional
from datetime import datetime, timedelta

# 导入聚宽内置的数据接口及基本面因子库
from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd


#############################################
# 1. 数据辅助类（DataHelper）
#############################################
class DataHelper:
    """
    数据操作辅助类，用于封装获取数据的接口调用
    主要包括：
    - get_price_safe: 获取指定股票的历史数据，捕获异常；
    - get_history_safe: 批量获取多只股票的历史数据。
    """

    @staticmethod
    def get_price_safe(
            security: Any,
            end_date: Any,
            frequency: str,
            fields: List[str],
            count: int,
            panel: bool = False,
            skip_paused: bool = True,
            fq: Optional[str] = None,
            fill_paused: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        安全调用 get_price 数据接口，获取指定股票或股票列表的历史数据。

        参数：
            security: 股票代码或股票代码列表。
            end_date: 数据截止日期。
            frequency: 数据频率，如 "daily" 或 "1m"。
            fields: 需要获取的字段列表，例如 ['open', 'close']。
            count: 请求数据的记录条数。
            panel: 是否返回面板数据，默认为False。
            skip_paused: 是否跳过停牌股票，默认为True。
            fq: 复权方式（例如 "pre" 或 "post"），默认为None。
            fill_paused: 是否补全停牌数据，默认为False。
        返回：
            返回包含数据的 DataFrame；出错则返回 None，并打印错误日志。
        """
        try:
            df = get_price(
                security,
                end_date=end_date,
                frequency=frequency,
                fields=fields,
                count=count,
                panel=panel,
                skip_paused=skip_paused,
                fq=fq,
                fill_paused=fill_paused
            )
            return df
        except Exception as e:
            log.error(f"获取 {security} 的价格数据时出错: {e}")
            return None

    @staticmethod
    def get_history_safe(
            security: Any,
            unit: str,
            field: str,
            count: int
    ) -> Optional[Dict[str, List[float]]]:
        """
        安全调用 history 数据接口，批量获取指定股票的历史数据。

        参数：
            security: 单只或多只股票代码。
            unit: 数据单位，比如 "1m" 表示1分钟数据。
            field: 需要获取的数据字段名称，如 "close"（收盘价）。
            count: 请求历史数据的条数。
        返回：
            返回一个字典，键为股票代码，值为对应数据列表；出错则返回 None。
        """
        try:
            data = history(count, unit=unit, field=field, security_list=security)
            return data
        except Exception as e:
            log.error(f"获取 {security} 的历史数据时出错: {e}")
            return None


#############################################
# 2. 交易策略类（TradingStrategy）
#############################################
class TradingStrategy:
    """
    交易策略核心类，封装了选股、调仓、买卖、止损及风控等功能。

    主要成员变量说明：
        - no_trading_today_signal：当天是否执行空仓操作（资金再平衡）。
        - hold_list：当前持仓股票列表。
        - target_list：本次调仓时筛选的目标股票列表。
        - not_buy_again：当天已买入股票列表，避免重复下单。
        - 以及其他参数，如止损策略、调仓股票数量等。
    """

    def __init__(self) -> None:
        # 基础策略控制变量
        self.no_trading_today_signal: bool = False  # 当天是否为空仓日（资金再平衡）
        self.pass_april: bool = True  # 是否在特定月份（如04月或01月）执行空仓策略
        self.run_stoploss: bool = True  # 是否启用止损策略

        # 持仓和调仓记录
        self.hold_list: List[str] = []  # 存储当前持仓股票代码
        self.yesterday_HL_list: List[str] = []  # 存储昨日涨停（收盘价==涨停价）的股票代码
        self.target_list: List[str] = []  # 本次调仓筛选出的目标股票代码
        self.not_buy_again: List[str] = []  # 当天已下单买入的股票代码，避免重复买入

        # 交易及风控参数
        self.stock_num: int = 9  # 目标持仓股票数量
        self.up_price: float = 100.0  # 过滤条件：排除价格高于此限的股票
        self.reason_to_sell: str = ''  # 记录卖出原因（如：limitup 或 stoploss）
        self.stoploss_strategy: int = 3  # 止损策略类型：1-个股止损，2-大盘止损，3-联合止损
        self.stoploss_limit: float = 0.88  # 个股止损阀值（成本价 * 0.88）
        self.stoploss_market: float = 0.94  # 大盘止损参数：大盘跌幅临界点

        # 成交量监控参数（是否启用及相关条件）
        self.HV_control: bool = False  # 是否启用异常成交量检测
        self.HV_duration: int = 120  # 参考过去多少天的成交量
        self.HV_ratio: float = 0.9  # 当日成交量大于历史最高成交量 * HV_ratio时视为异常

        # 状态机字典，用于记录交易信号和风险水平
        self.state: Dict[str, Any] = {
            'buy_signal': False,
            'sell_signal': False,
            'risk_level': 'normal'
        }

    def initialize(self, context: Any) -> None:
        """
        策略初始化函数——配置交易环境，包括：
            - 启用防未来数据；
            - 设置策略基准（如上证指数）；
            - 设置真实市场价格和固定滑点；
            - 设置订单成本（如佣金和印花税）。
        参数：
            context: 聚宽平台传入的交易上下文对象。
        """
        set_option('avoid_future_data', True)
        set_benchmark('000001.XSHG')
        set_option('use_real_price', True)
        # 设置固定滑点，固定为3/10000
        set_slippage(FixedSlippage(3 / 10000))
        # 设置订单成本参数：印花税、佣金、最低佣金等
        set_order_cost(OrderCost(
            open_tax=0,
            close_tax=0.001,  # 卖出时印花税 0.1%
            open_commission=2.5 / 10000,
            close_commission=2.5 / 10000,
            close_today_commission=0,
            min_commission=5  # 最低佣金5元
        ), type='stock')
        # 设置日志输出等级（只输出错误和调试信息）
        log.set_level('order', 'error')
        log.set_level('system', 'error')
        log.set_level('strategy', 'debug')

    def check_holdings_yesterday(self, context: Any) -> None:
        """
        检查并打印当前持仓股票在昨日的交易数据（开盘价、收盘价及涨跌幅）。
        参数：
            context: 聚宽平台传入的交易上下文对象。
        """
        positions = context.portfolio.positions
        if not positions:
            log.info("昨日没有持仓数据。")
            return

        log.info("== 昨日持仓股票交易数据 ==")
        for stock, position in positions.items():
            try:
                df = DataHelper.get_price_safe(
                    stock,
                    end_date=context.previous_date,
                    frequency="daily",
                    fields=['open', 'close'],
                    count=1,
                    panel=False
                )
                if df is None or df.empty:
                    log.info(f"无法获取股票 {stock} 的昨日数据。")
                    continue
                open_price: float = df.iloc[0]['open']
                close_price: float = df.iloc[0]['close']
                change_pct: float = (close_price / open_price - 1) * 100
                log.info(f"股票 {stock}：持仓 {position.total_amount} 股，"
                         f"开盘价 {open_price:.2f}，收盘价 {close_price:.2f}，"
                         f"涨跌幅 {change_pct:.2f}%")
            except Exception as e:
                log.error(f"处理股票 {stock} 数据时出错: {e}")

    def prepare_stock_list(self, context: Any) -> None:
        """
        更新持仓列表并筛选昨日涨停股票，同时判断是否为资金再平衡（空仓）日。
        参数：
            context: 聚宽平台传入的交易上下文对象。
        """
        # 更新当前持仓股票列表
        self.hold_list = [position.security for position in list(context.portfolio.positions.values())]
        if self.hold_list:
            # 获取昨日持仓股票的收盘价、涨停价和跌停价
            df = DataHelper.get_price_safe(
                self.hold_list,
                end_date=context.previous_date,
                frequency='daily',
                fields=['close', 'high_limit', 'low_limit'],
                count=1,
                panel=False,
                fill_paused=False
            )
            if df is not None and not df.empty:
                # 收盘价等于涨停价的股票视为昨日涨停股票
                self.yesterday_HL_list = list(df[df['close'] == df['high_limit']]['code'])
            else:
                self.yesterday_HL_list = []
        else:
            self.yesterday_HL_list = []

        # 判断当前是否为空仓日（例如04月或01月），返回True为资金再平衡日
        self.no_trading_today_signal = self.today_is_between(context)

    def get_stock_list(self, context: Any) -> List[str]:
        """
        选股模块：
         1. 从指定指数（如 399101.XSHE）中获取初始股票池；
         2. 依次过滤：次新股、科创股/北交股、ST风险股、停牌股、当日涨停或跌停股票；
         3. 根据基本面数据（EPS、市值）排序后，返回候选股票列表。

        参数：
            context: 交易上下文对象。
        返回：
            筛选后的候选股票代码列表。
        """
        # 从指定指数（399101.XSHE）获取初步股票列表
        MKT_index: str = '399101.XSHE'
        initial_list: List[str] = get_index_stocks(MKT_index)

        # 依次应用多个过滤器，剔除不符合条件的股票
        initial_list = self.filter_new_stock(context, initial_list)  # 过滤次新股（上市不足375天）
        initial_list = self.filter_kcbj_stock(initial_list)  # 过滤科创或北交股票
        initial_list = self.filter_st_stock(initial_list)  # 过滤ST或风险股票
        initial_list = self.filter_paused_stock(initial_list)  # 过滤停牌股票
        initial_list = self.filter_limitup_stock(context, initial_list)  # 过滤当日涨停股票（未持仓情况下）
        initial_list = self.filter_limitdown_stock(context, initial_list)  # 过滤当日跌停股票（未持仓情况下）

        # 根据基本面数据进行排序：利用EPS、市值等因子筛选
        q = query(valuation.code, indicator.eps) \
            .filter(valuation.code.in_(initial_list)) \
            .order_by(valuation.market_cap.asc())
        df = get_fundamentals(q)
        stock_list: List[str] = list(df.code)
        # 限制候选股票数量，避免处理数据过大
        stock_list = stock_list[:50]
        # 取前2倍目标持仓数作为候选池
        final_list: List[str] = stock_list[:2 * self.stock_num]
        log.info(f"初选候选股票: {final_list}")

        # 打印候选股票的基本面信息，方便验证筛选逻辑
        if final_list:
            info_query = query(
                valuation.code,
                income.pubDate,
                income.statDate,
                income.operating_revenue,
                indicator.eps
            ).filter(valuation.code.in_(final_list))
            df_info = get_fundamentals(info_query)
            for _, row in df_info.iterrows():
                log.info(f"股票 {row['code']}：报告日期 {row.get('pubDate', 'N/A')}，"
                         f"统计日期 {row.get('statDate', 'N/A')}，营业收入 {row.get('operating_revenue', 'N/A')}，"
                         f"EPS {row.get('eps', 'N/A')}")
        return final_list

    def weekly_adjustment(self, context: Any) -> None:
        """
        每周调仓策略：
         1. 如果今天不是空仓日，则选股生成目标股票列表；
         2. 遍历当前持仓，卖出不在目标列表内且昨日未涨停的股票；
         3. 对目标股票进行买入操作，并记录买入以避免重复下单。

        参数：
            context: 交易上下文对象。
        """
        if not self.no_trading_today_signal:
            # 重置当天已买入记录
            self.not_buy_again = []
            # 根据选股函数获取目标股票列表
            self.target_list = self.get_stock_list(context)
            target_list: List[str] = self.target_list[:self.stock_num]
            log.info(f"每周调仓目标股票: {target_list}")

            # 遍历当前持仓，卖出不在目标列表且昨日未涨停的股票
            for stock in self.hold_list:
                if stock not in target_list and stock not in self.yesterday_HL_list:
                    log.info(f"卖出股票 {stock}")
                    position = context.portfolio.positions[stock]
                    self.close_position(position)
                else:
                    log.info(f"继续持有股票 {stock}")

            # 对目标股票进行买入操作
            self.buy_security(context, target_list)

            # 更新当天买入记录，防止重复下单
            for position in list(context.portfolio.positions.values()):
                if position.security not in self.not_buy_again:
                    self.not_buy_again.append(position.security)

    def check_limit_up(self, context: Any) -> None:
        """
        检查昨日涨停的股票是否破板（当前价格低于涨停价）。
        如破板，则立即卖出，并记录卖出原因 'limitup'。

        参数：
            context: 交易上下文对象。
        """
        now_time = context.current_dt
        if self.yesterday_HL_list:
            for stock in self.yesterday_HL_list:
                current_data = DataHelper.get_price_safe(
                    stock,
                    end_date=now_time,
                    frequency='1m',
                    fields=['close', 'high_limit'],
                    count=1,
                    panel=False,
                    fill_paused=True
                )
                if current_data is not None and not current_data.empty:
                    if current_data.iloc[0]['close'] < current_data.iloc[0]['high_limit']:
                        log.info(f"股票 {stock} 涨停破板，执行卖出。")
                        position = context.portfolio.positions[stock]
                        self.close_position(position)
                        self.reason_to_sell = 'limitup'
                    else:
                        log.info(f"股票 {stock} 收盘仍维持涨停状态。")

    def check_remain_amount(self, context: Any) -> None:
        """
        检查当前持仓数量是否达标（若因涨停破板导致持仓不足）。
        如果持仓不足，选出未买入的目标股票进行补仓操作。

        参数：
            context: 交易上下文对象。
        """
        if self.reason_to_sell == 'limitup':
            self.hold_list = [position.security for position in list(context.portfolio.positions.values())]
            if len(self.hold_list) < self.stock_num:
                target_list = self.filter_not_buy_again(self.target_list)
                target_list = target_list[:min(self.stock_num, len(target_list))]
                log.info(f"补仓需求：可用资金 {round(context.portfolio.cash, 2)}，候选补仓股票: {target_list}")
                self.buy_security(context, target_list)
            self.reason_to_sell = ''
        else:
            log.info("未检测到需要补仓的情况。")

    def trade_afternoon(self, context: Any) -> None:
        """
        下午交易任务流程：
         1. 检查涨停破板触发的卖出信号；
         2. 如果启用了成交量监测，则检查异常成交量；
         3. 检查是否需要补仓操作。

        参数：
            context: 交易上下文对象。
        """
        if not self.no_trading_today_signal:
            self.check_limit_up(context)
            if self.HV_control:
                self.check_high_volume(context)
            self.check_remain_amount(context)

    def sell_stocks(self, context: Any) -> None:
        """
        止盈和止损操作：
         判断是否执行卖出操作（个股止损、大盘止损或联合止损），
         并调用平仓函数卖出全部或部分持仓股票。

        参数：
            context: 交易上下文对象。
        """
        if self.run_stoploss:
            if self.stoploss_strategy == 1:
                # 个股止盈或止损：当盈利达到一定比例或亏损触及止损线时直接卖出
                for stock in list(context.portfolio.positions.keys()):
                    pos = context.portfolio.positions[stock]
                    if pos.price >= pos.avg_cost * 2:
                        order_target_value(stock, 0)
                        log.debug(f"股票 {stock} 盈利达100%，执行止盈卖出。")
                    elif pos.price < pos.avg_cost * self.stoploss_limit:
                        order_target_value(stock, 0)
                        log.debug(f"股票 {stock} 触及止损线，执行卖出。")
                        self.reason_to_sell = 'stoploss'
            elif self.stoploss_strategy == 2:
                # 大盘止损：检查大盘跌幅，若整体跌幅超过阈值则全部平仓
                stock_list = get_index_stocks('399101.XSHE')
                df = DataHelper.get_price_safe(
                    stock_list,
                    end_date=context.previous_date,
                    frequency='daily',
                    fields=['close', 'open'],
                    count=1,
                    panel=False
                )
                if df is not None and not df.empty:
                    down_ratio = (df['close'] / df['open']).mean()
                    if down_ratio <= self.stoploss_market:
                        self.reason_to_sell = 'stoploss'
                        log.debug(f"大盘跌幅达到 {down_ratio:.2%}，执行平仓操作。")
                        for stock in list(context.portfolio.positions.keys()):
                            order_target_value(stock, 0)
            elif self.stoploss_strategy == 3:
                # 联合止损：结合大盘及个股情况进行止损判断
                stock_list = get_index_stocks('399101.XSHE')
                df = DataHelper.get_price_safe(
                    stock_list,
                    end_date=context.previous_date,
                    frequency='daily',
                    fields=['close', 'open'],
                    count=1,
                    panel=False
                )
                if df is not None and not df.empty:
                    down_ratio = (df['close'] / df['open']).mean()
                    if down_ratio <= self.stoploss_market:
                        self.reason_to_sell = 'stoploss'
                        log.debug(f"大盘跌幅达到 {down_ratio:.2%}，执行平仓操作。")
                        for stock in list(context.portfolio.positions.keys()):
                            order_target_value(stock, 0)
                    else:
                        for stock in list(context.portfolio.positions.keys()):
                            pos = context.portfolio.positions[stock]
                            if pos.price < pos.avg_cost * self.stoploss_limit:
                                order_target_value(stock, 0)
                                log.debug(f"股票 {stock} 触及止损，执行卖出。")
                                self.reason_to_sell = 'stoploss'

    def check_high_volume(self, context: Any) -> None:
        """
        检查持仓股票当天成交量是否异常（高于过去 HV_duration 天中的最高成交量的 HV_ratio 倍）。
        若发现异常，则执行平仓操作。

        参数：
            context: 交易上下文对象。
        """
        current_data = get_current_data()
        for stock in list(context.portfolio.positions.keys()):
            if current_data[stock].paused:
                continue
            if current_data[stock].last_price == current_data[stock].high_limit:
                continue
            if context.portfolio.positions[stock].closeable_amount == 0:
                continue
            df_volume = get_bars(
                stock,
                count=self.HV_duration,
                unit='1d',
                fields=['volume'],
                include_now=True,
                df=True
            )
            if df_volume is not None and not df_volume.empty:
                if df_volume['volume'].iloc[-1] > self.HV_ratio * df_volume['volume'].max():
                    log.info(f"检测到 {stock} 异常放量，执行卖出操作。")
                    position = context.portfolio.positions[stock]
                    self.close_position(position)

    # ----- 以下为股票过滤器函数，依次过滤不符合条件的股票 -----

    def filter_paused_stock(self, stock_list: List[str]) -> List[str]:
        """
        过滤停牌股票
        """
        current_data = get_current_data()
        return [stock for stock in stock_list if not current_data[stock].paused]

    def filter_st_stock(self, stock_list: List[str]) -> List[str]:
        """
        过滤含有 ST 或风险标识的股票
        """
        current_data = get_current_data()
        return [stock for stock in stock_list if (not current_data[stock].is_st) and
                ('ST' not in current_data[stock].name) and
                ('*' not in current_data[stock].name) and
                ('退' not in current_data[stock].name)]

    def filter_kcbj_stock(self, stock_list: List[str]) -> List[str]:
        """
        过滤科创、北交股票（股票代码以 '4'、'8' 开头或 '68' 开头的剔除）
        """
        return [stock for stock in stock_list if stock[0] not in ('4', '8') and not stock.startswith('68')]

    def filter_limitup_stock(self, context: Any, stock_list: List[str]) -> List[str]:
        """
        过滤当天已涨停的股票（若未持仓则过滤）。
        """
        history_data = DataHelper.get_history_safe(stock_list, unit='1m', field='close', count=1)
        current_data = get_current_data()
        if history_data is None:
            return stock_list
        return [stock for stock in stock_list if stock in context.portfolio.positions.keys() or
                (history_data.get(stock, [0])[-1] < current_data[stock].high_limit)]

    def filter_limitdown_stock(self, context: Any, stock_list: List[str]) -> List[str]:
        """
        过滤当天已跌停的股票（若未持仓则过滤）。
        """
        history_data = DataHelper.get_history_safe(stock_list, unit='1m', field='close', count=1)
        current_data = get_current_data()
        if history_data is None:
            return stock_list
        return [stock for stock in stock_list if stock in context.portfolio.positions.keys() or
                (history_data.get(stock, [float('inf')])[-1] > current_data[stock].low_limit)]

    def filter_new_stock(self, context: Any, stock_list: List[str]) -> List[str]:
        """
        过滤次新股：排除上市不足375天的股票。
        """
        yesterday = context.previous_date
        return [stock for stock in stock_list if
                not (yesterday - get_security_info(stock).start_date < timedelta(days=375))]

    def filter_highprice_stock(self, context: Any, stock_list: List[str]) -> List[str]:
        """
        过滤股价高于 up_price 的股票（非持仓股票）。
        """
        history_data = DataHelper.get_history_safe(stock_list, unit='1m', field='close', count=1)
        if history_data is None:
            return stock_list
        return [stock for stock in stock_list if stock in context.portfolio.positions.keys() or
                history_data.get(stock, [self.up_price + 1])[-1] <= self.up_price]

    def filter_not_buy_again(self, stock_list: List[str]) -> List[str]:
        """
        过滤当天已买入的股票，防止重复下单。
        """
        return [stock for stock in stock_list if stock not in self.not_buy_again]

    # ----- 下单及仓位管理函数 -----

    def order_target_value_(self, security: str, value: float) -> Any:
        """
        封装 order_target_value 函数，统一下单操作并记录日志。

        参数：
            security: 股票代码。
            value: 下单目标资金数额。
        返回：
            返回订单对象；出错则返回 None。
        """
        if value != 0:
            log.debug(f"正在为 {security} 下单，目标金额 {value}")
        try:
            order = order_target_value(security, value)
            return order
        except Exception as e:
            log.error(f"股票 {security} 下单时出错，目标金额 {value}，错误信息: {e}")
            return None

    def open_position(self, security: str, value: float) -> bool:
        """
        执行买入操作（开仓），按分配资金买入股票。

        参数：
            security: 股票代码。
            value: 分配的资金数额。
        返回：
            如果买入成功返回 True，否则返回 False。
        """
        order = self.order_target_value_(security, value)
        if order is not None and order.filled > 0:
            return True
        return False

    def close_position(self, position: Any) -> bool:
        """
        执行平仓操作，尽可能将持仓全部卖出。

        参数：
            position: 持仓对象。
        返回：
            如果订单全部成交返回 True，否则返回 False。
        """
        security = position.security
        order = self.order_target_value_(security, 0)
        if order is not None and order.status == OrderStatus.held and order.filled == order.amount:
            return True
        return False

    def buy_security(self, context: Any, target_list: List[str]) -> None:
        """
        买入操作：对目标股票执行买入，下单资金均摊分配。

        参数：
            context: 交易上下文对象。
            target_list: 目标股票代码列表。
        """
        position_count = len(context.portfolio.positions)
        target_num = len(target_list)
        if target_num > position_count:
            try:
                # 计算每只股票分配的资金：账户剩余现金 / (目标数量 - 当前持仓数量)
                value = context.portfolio.cash / (target_num - position_count)
            except ZeroDivisionError as e:
                log.error(f"资金分摊时除零错误: {e}")
                return
            for stock in target_list:
                # 如果该股票未持仓，则执行买入
                if context.portfolio.positions[stock].total_amount == 0:
                    if self.open_position(stock, value):
                        log.info(f"已买入股票 {stock}，分配资金 {value:.2f}")
                        self.not_buy_again.append(stock)
                        if len(context.portfolio.positions) == target_num:
                            break

    def today_is_between(self, context: Any) -> bool:
        """
        判断当前日期是否为资金再平衡（空仓）日。
        通常在04月或01月期间会执行空仓操作。

        参数：
            context: 交易上下文对象。
        返回：
            True 表示当天为空仓日，否则为 False。
        """
        today_str = context.current_dt.strftime('%m-%d')
        # 当设定 pass_april 为 True 时，04月份和01月份为空仓日
        if self.pass_april:
            if ('04-01' <= today_str <= '04-30') or ('01-01' <= today_str <= '01-30'):
                return True
            else:
                return False
        else:
            return False

    def close_account(self, context: Any) -> None:
        """
        空仓日清仓：如果当天为空仓日，则平仓所有持仓股票。

        参数：
            context: 交易上下文对象。
        """
        if self.no_trading_today_signal:
            if self.hold_list:
                for stock in self.hold_list:
                    position = context.portfolio.positions[stock]
                    self.close_position(position)
                    log.info(f"空仓日平仓，卖出股票 {stock}。")

    def print_position_info(self, context: Any) -> None:
        """
        打印持仓详细信息，包括股票代码、成本、现价、涨跌幅、持仓股数与市值。

        参数：
            context: 交易上下文对象。
        """
        for position in list(context.portfolio.positions.values()):
            securities: str = position.security
            cost: float = position.avg_cost
            price: float = position.price
            ret: float = 100 * (price / cost - 1)
            value: float = position.value
            amount: int = position.total_amount
            print(f"股票: {securities}")
            print(f"成本价: {cost:.2f}")
            print(f"现价: {price:.2f}")
            print(f"涨跌幅: {ret:.2f}%")
            print(f"持仓: {amount}")
            print(f"市值: {value:.2f}")
            print("--------------------------------------")
        print("********** 持仓信息打印结束 **********")


#############################################
# 3. 全局包装函数（调度任务入口）
#############################################

def prepare_stock_list_func(context: Any) -> None:
    """包装调用 prepare_stock_list 函数，用于每天开盘前更新持仓和候选股票列表"""
    strategy.prepare_stock_list(context)


def check_holdings_yesterday_func(context: Any) -> None:
    """包装调用 check_holdings_yesterday 函数，检查持仓股票昨日数据"""
    strategy.check_holdings_yesterday(context)


def weekly_adjustment_func(context: Any) -> None:
    """包装调用 weekly_adjustment 函数，用于每周调仓操作"""
    strategy.weekly_adjustment(context)


def sell_stocks_func(context: Any) -> None:
    """包装调用 sell_stocks 函数，用于每日卖出操作（止盈/止损）"""
    strategy.sell_stocks(context)


def trade_afternoon_func(context: Any) -> None:
    """包装调用 trade_afternoon 函数，用于下午的交易任务（补仓、卖出检查等）"""
    strategy.trade_afternoon(context)


def close_account_func(context: Any) -> None:
    """包装调用 close_account 函数，用于空仓日清仓操作"""
    strategy.close_account(context)


def print_position_info_func(context: Any) -> None:
    """包装调用 print_position_info 函数，用于每周打印持仓信息"""
    strategy.print_position_info(context)


#############################################
# 4. 初始化及全局策略设置（入口函数）
#############################################

# 创建全局策略实例，供全局任务调用
strategy = TradingStrategy()

# 2024-12-05	09:31:00	中电港(001287.XSHE)
# 2024-09-12	09:31:00	鼎龙科技(603004.XSHG)
def initialize(context: Any) -> None:
    """
    聚宽平台全局初始化函数：
      1. 调用策略的 initialize 方法，配置交易环境参数；
      2. 注册各个调度任务。注意：所有任务均使用顶层函数，避免序列化问题。

    参数：
        context: 交易上下文对象（由聚宽平台传入）。
    """
    # 初始化策略环境（设置防未来数据、基准、滑点、订单成本等）
    strategy.initialize(context)

    # 注册调度任务：
    # 每天开盘前更新持仓和股票列表，检查昨日持仓数据
    run_daily(prepare_stock_list_func, time='9:05')
    run_daily(check_holdings_yesterday_func, time='9:00')
    # 每周在星期二执行调仓
    run_weekly(weekly_adjustment_func, 2, time='10:30')
    # 每天上午执行止盈止损检查
    run_daily(sell_stocks_func, time='10:00')
    # 每天下午执行补仓、成交量检查和其他任务
    run_daily(trade_afternoon_func, time='14:30')
    # 每天下午近尾市时（14:50）检查空仓操作
    run_daily(close_account_func, time='14:50')
    # 每周在星期五尾市时输出持仓信息
    run_weekly(print_position_info_func, 5, time='15:10')