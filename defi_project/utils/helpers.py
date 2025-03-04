"""
Helper Functions Module
Provides commonly used utility functions
"""
import json
import logging
import os
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import hashlib
import random
import string

# Set up logging
logger = logging.getLogger(__name__)

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file
    
    Args:
        file_path: File path
        
    Returns:
        JSON data dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        return {}

def save_json_file(file_path: str, data: Dict[str, Any]) -> bool:
    """
    Save JSON file
    
    Args:
        file_path: File path
        data: Data to save
        
    Returns:
        Whether save was successful
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        return False

def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
    """
    Convert timestamp to datetime
    
    Args:
        timestamp: Timestamp (seconds)
        
    Returns:
        Datetime object
    """
    return datetime.fromtimestamp(timestamp)

def datetime_to_timestamp(dt: datetime) -> int:
    """
    Convert datetime to timestamp
    
    Args:
        dt: Datetime object
        
    Returns:
        Timestamp (seconds)
    """
    return int(dt.timestamp())

def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime
    
    Args:
        dt: Datetime object
        format_str: Format string
        
    Returns:
        Formatted datetime string
    """
    return dt.strftime(format_str)

def parse_datetime(datetime_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """
    Parse datetime string
    
    Args:
        datetime_str: Datetime string
        format_str: Format string
        
    Returns:
        Datetime object
    """
    return datetime.strptime(datetime_str, format_str)

def generate_random_string(length: int = 10) -> str:
    """
    生成随机字符串
    
    Args:
        length: 字符串长度
        
    Returns:
        随机字符串
    """
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def generate_hash(data: str) -> str:
    """
    生成哈希值
    
    Args:
        data: 要哈希的数据
        
    Returns:
        哈希值
    """
    return hashlib.sha256(data.encode()).hexdigest()

def format_number(number: float, decimal_places: int = 2) -> str:
    """
    格式化数字
    
    Args:
        number: 要格式化的数字
        decimal_places: 小数位数
        
    Returns:
        格式化后的数字字符串
    """
    format_str = f"{{:.{decimal_places}f}}"
    return format_str.format(number)

def format_currency(amount: float, currency: str = "USD", decimal_places: int = 2) -> str:
    """
    格式化货币
    
    Args:
        amount: 金额
        currency: 货币类型
        decimal_places: 小数位数
        
    Returns:
        格式化后的货币字符串
    """
    formatted_amount = format_number(amount, decimal_places)
    
    if currency == "USD":
        return f"${formatted_amount}"
    elif currency == "ETH":
        return f"{formatted_amount} ETH"
    elif currency == "BTC":
        return f"{formatted_amount} BTC"
    else:
        return f"{formatted_amount} {currency}"

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    格式化百分比
    
    Args:
        value: 值(0-1)
        decimal_places: 小数位数
        
    Returns:
        格式化后的百分比字符串
    """
    percentage = value * 100
    return f"{format_number(percentage, decimal_places)}%"

def calculate_time_difference(start_time: datetime, end_time: datetime = None) -> Dict[str, Any]:
    """
    计算时间差
    
    Args:
        start_time: 开始时间
        end_time: 结束时间，默认为当前时间
        
    Returns:
        时间差字典
    """
    if end_time is None:
        end_time = datetime.now()
    
    diff = end_time - start_time
    
    days = diff.days
    hours, remainder = divmod(diff.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return {
        'days': days,
        'hours': hours,
        'minutes': minutes,
        'seconds': seconds,
        'total_seconds': diff.total_seconds()
    }

def format_time_difference(time_diff: Dict[str, Any]) -> str:
    """
    格式化时间差
    
    Args:
        time_diff: 时间差字典
        
    Returns:
        格式化后的时间差字符串
    """
    days = time_diff['days']
    hours = time_diff['hours']
    minutes = time_diff['minutes']
    seconds = time_diff['seconds']
    
    if days > 0:
        return f"{days}天 {hours}小时 {minutes}分钟"
    elif hours > 0:
        return f"{hours}小时 {minutes}分钟"
    elif minutes > 0:
        return f"{minutes}分钟 {seconds}秒"
    else:
        return f"{seconds}秒"

def is_valid_ethereum_address(address: str) -> bool:
    """
    检查是否为有效的以太坊地址
    
    Args:
        address: 以太坊地址
        
    Returns:
        是否有效
    """
    if not address.startswith('0x'):
        return False
    
    if len(address) != 42:  # '0x' + 40个十六进制字符
        return False
    
    # 检查是否只包含十六进制字符
    try:
        int(address[2:], 16)
        return True
    except ValueError:
        return False

def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    重试装饰器
    
    Args:
        max_attempts: 最大尝试次数
        delay: 延迟时间(秒)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise
                    logger.warning(f"函数 {func.__name__} 执行失败，正在重试({attempts}/{max_attempts}): {str(e)}")
                    time.sleep(delay)
        return wrapper
    return decorator

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    将列表分块
    
    Args:
        lst: 要分块的列表
        chunk_size: 块大小
        
    Returns:
        分块后的列表
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def flatten_list(lst: List[List[Any]]) -> List[Any]:
    """
    扁平化嵌套列表
    
    Args:
        lst: 嵌套列表
        
    Returns:
        扁平化后的列表
    """
    return [item for sublist in lst for item in sublist]

def deep_get(dictionary: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    从嵌套字典中获取值
    
    Args:
        dictionary: 字典
        keys: 键列表
        default: 默认值
        
    Returns:
        获取的值
    """
    if not dictionary:
        return default
    
    if not keys:
        return dictionary
    
    current = dictionary
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current

def deep_set(dictionary: Dict[str, Any], keys: List[str], value: Any) -> Dict[str, Any]:
    """
    在嵌套字典中设置值
    
    Args:
        dictionary: 字典
        keys: 键列表
        value: 要设置的值
        
    Returns:
        更新后的字典
    """
    if not keys:
        return dictionary
    
    current = dictionary
    for i, key in enumerate(keys[:-1]):
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
    return dictionary

def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并两个字典
    
    Args:
        dict1: 第一个字典
        dict2: 第二个字典
        
    Returns:
        合并后的字典
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result 