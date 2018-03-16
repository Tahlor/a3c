
import datetime
import dateutil.parser

def getDateTimeFromISO8601String(s):
    d = dateutil.parser.parse(s)
    return d
