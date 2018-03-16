
import datetime
import dateutil.parser
from datetime import timezone

def getDateTimeFromISO8601String(s):
    d = dateutil.parser.parse(s)
    seconds = (d - datetime.datetime(2010, 1, 1, tzinfo=timezone.utc)).total_seconds()
    return seconds

def buy_sell_encoder(s):
    #s = s.decode()
    if s == b"buy":
        return 1
    elif s == b"sell":
        return 0
    else:
        return -1

def round_to_nearest(number, round_by):
    return int(number) - (int(number) % round_by)