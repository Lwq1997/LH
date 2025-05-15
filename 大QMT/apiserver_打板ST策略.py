# 研究环境文件，只需要把123456替换为自己的授权码，可以使用ctrl+F查找，别处无需改动。授权码请联系微信xjip20
import requests
import json
import pandas as pd


class JoinquantTrader:
    def __init__(self, url_list=['http://106.54.211.231--3333--123456']):  # 把123456替换为自己的授权码，没有授权码联系微信xjip20
        self.url_list = url_list

    def get_user_data(self, data_type='用户信息'):
        url_list = self.url_list
        for url_str in url_list:
            parts = url_str.split('--')
            if len(parts) == 3:
                url, port, password = parts
                print(f"get_user_data的URL: {url}, 端口: {port}, 密码: {password}")
                url = '{}:{}/_dash-update-component'.format(url, port)
                headers = {'Content-Type': 'application/json'}
                data = {
                    "output": "joinquant_trader_table.data@63d85b6189e42cba63feea36381da615c31ad8e36ae420ed67f60f3598efc9ad",
                    "outputs": {"id": "joinquant_trader_table",
                                "property": "data@63d85b6189e42cba63feea36381da615c31ad8e36ae420ed67f60f3598efc9ad"},
                    "inputs": [{"id": "joinquant_trader_password", "property": "value", "value": password},
                               {"id": "joinquant_trader_data_type", "property": "value", "value": data_type},
                               {"id": "joinquant_trader_text", "property": "value",
                                "value": "\n               {'状态': 'held', '订单添加时间': 'datetime.datetime(2024, 4, 23, 9, 30)', '买卖': 'False', '下单数量': '9400', '已经成交': '9400', '股票代码': '001.XSHE', '订单ID': '1732208241', '平均成交价格': '10.5', '持仓成本': '10.59', '多空': 'long', '交易费用': '128.31'}\n                "},
                               {"id": "joinquant_trader_run", "property": "value", "value": "运行"},
                               {"id": "joinquant_trader_down_data", "property": "value", "value": "不下载数据"}],
                    "changedPropIds": ["joinquant_trader_run.value"],
                    "parsedChangedPropsIds": ["joinquant_trader_run.value"]}
                res = requests.post(url=url, data=json.dumps(data), headers=headers)
                text = res.json()
                df = pd.DataFrame(text['response']['joinquant_trader_table']['data'])
            else:
                print(f"格式异常: {url_str}")

    def send_order(self, result):
        url_list = self.url_list
        for url_str in url_list:
            parts = url_str.split('--')
            if len(parts) == 3:
                url, port, password = parts
                print(f"send_order的URL: {url}, 端口: {port}, 密码: {password}")
                url = '{}:{}/_dash-update-component'.format(url, port)
                headers = {'Content-Type': 'application/json'}
                data = {
                    "output": "joinquant_trader_table.data@63d85b6189e42cba63feea36381da615c31ad8e36ae420ed67f60f3598efc9ad",
                    "outputs": {"id": "joinquant_trader_table",
                                "property": "data@63d85b6189e42cba63feea36381da615c31ad8e36ae420ed67f60f3598efc9ad"},
                    "inputs": [{"id": "joinquant_trader_password", "property": "value", "value": password},
                               {"id": "joinquant_trader_data_type", "property": "value", "value": '实时数据'},
                               {"id": "joinquant_trader_text", "property": "value", "value": result},
                               {"id": "joinquant_trader_run", "property": "value", "value": "运行"},
                               {"id": "joinquant_trader_down_data", "property": "value", "value": "不下载数据"}],
                    "changedPropIds": ["joinquant_trader_run.value"],
                    "parsedChangedPropsIds": ["joinquant_trader_run.value"]}
                res = requests.post(url=url, data=json.dumps(data), headers=headers)
                text = res.json()
                df = pd.DataFrame(text['response']['joinquant_trader_table']['data'])
            else:
                print(f"格式异常: {url_str}")


def send_order(result):
    url_list = ['http://server.588gs.cn--2000--打板ST策略',
                'http://106.54.211.231--3333--打板ST策略']
    # 把123456替换为自己的授权码，没有授权码联系微信xjip20
    api_data = JoinquantTrader(url_list=url_list)
    data = {}
    data['状态'] = str(result.status)
    data['订单添加时间'] = str(result.add_time)
    data['买卖'] = str(result.is_buy)
    data['下单数量'] = str(result.amount)
    data['已经成交'] = str(result.filled)
    data['股票代码'] = str(result.security)
    data['订单ID'] = str(result.order_id)
    data['平均成交价格'] = str(result.price)
    data['持仓成本'] = str(result.avg_cost)
    data['多空'] = str(result.side)
    data['交易费用'] = str(result.commission)
    result = str(data)
    api_data.send_order(result)
    return data


def api_order(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is None:
            return
        send_order(result)
        return result

    return wrapper


def api_order_target(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is None:
            return
        send_order(result)
        return result

    return wrapper


def api_order_value(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is None:
            return
        send_order(result)
        return result

    return wrapper


def api_order_target_value(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is None:
            return
        send_order(result)
        return result

    return wrapper
