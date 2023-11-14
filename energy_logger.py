import time
from urllib.request import urlopen
import json
import threading
import csv  

class Logger(threading.Thread):
    def __init__(self, start_time, url, metrics, target) :
        super(Logger, self).__init__()
        self.daemon = True 
        self.start_time=start_time
        self.is_logging_active=True
        self.url=url
        self.metrics=metrics
        self.target=target
        self.last_read_metrics = []

        # http://localhost:9090/api/v1/query?query={__name__=~%22energon_cpu_in_power_mW|energon_gpu_in_power_mW|energon_cpu_total_usage_percentage|energon_gpu_total_usage_percentage|energon_ram_used_percentage|furcifer_latency_ms%22,%20instance=~%22localhost:9877%22}
        self.query_url = url + "{" + "__name__=~'{}'".format("|".join(metrics)) + ",instance='{}'".format(target) + "}"

    def set_is_logging_active(self, is_logging_active):
        self.is_logging_active = is_logging_active

    def find_metric(self, obj, metric):
        result = {}
        if len(obj["data"]["result"]) == 0:
            return result
        for m in obj["data"]["result"]:
            if m["metric"]["__name__"] == metric:
                result["timestamp"] = m["value"][0]
                result["value"] = m["value"][1]
                break
        return result

    def run(self):
        while(True):
            if self.is_logging_active == True:
                response = urlopen(self.query_url)
                data_json = json.loads(response.read())
                last_timestamp = -1
                self.last_read_metrics = []
                for metric in self.metrics:
                    try:
                        metric_query_result_firtered = self.find_metric(data_json, metric)
                        print("-----------", metric)
                        print("data_json", metric_query_result_firtered)
                        value = metric_query_result_firtered["value"]
                        last_timestamp = metric_query_result_firtered["timestamp"]
                    except Exception as e:
                        print("Couldn't get the data, please check the server", e)
                        value = -1
                    self.last_read_metrics += [value]
                self.last_read_metrics += [last_timestamp]
                print("self.last_read_metrics", self.last_read_metrics)
                time.sleep(1)

    def get_metrics(self):
        return self.last_read_metrics   
    
if __name__ == "__main__":
    IP_PROMETHEUS_SERVER = "localhost"
    PORT_PROMETHEUS_SERVER = "9090"

    # raspberry pi 4
    # TARGET = "128.195.55.248:8000"
    TARGET = "192.168.1.30:9877"
    
    # jetson nano
    # TARGET = "128.195.55.244:9877"
    # TARGET = "192.168.1.10:9877"

    # jetson orin
    # TARGET = "128.195.55.253:9877"

    # jetson xavier
    # TARGET = "192.168.1.29:9877"
    # TARGET = "192.168.1.29:9877"

    metrics = []
    metrics.append("energon_total_in_power_mW")
    metrics.append("energon_cpu_in_power_mW")
    metrics.append("energon_gpu_in_power_mW")
    metrics.append("energon_cpu_total_usage_percentage")
    metrics.append("energon_gpu_total_usage_percentage")
    metrics.append("energon_ram_used_percentage")

    log_metrics = Logger(time.time(), "http://{}:{}/api/v1/query?query=".format(IP_PROMETHEUS_SERVER, PORT_PROMETHEUS_SERVER), metrics, TARGET)
    log_metrics.start()

    header = metrics + ["timestamp"]

    with open('./reports/{}.csv'.format(TARGET),'a') as fd:
        fd.write(",".join(header) + "\n")

    while(True):
        time.sleep(0.5)
        metrics = log_metrics.get_metrics()
        if len(metrics) > 0:
            with open('./reports/{}.csv'.format(TARGET),'a') as fd:
                fd.write(",".join([str(m) for m in metrics]) + "\n")