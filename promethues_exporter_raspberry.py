from bluetooth import *
import datetime
from prometheus_client import start_http_server, Gauge, Info
import time

class UM25C(object):
    def __init__(self, device_address, polling_interval_seconds, energon_port):
        self.device_address = device_address
        self.polling_interval_seconds = polling_interval_seconds
        self.energon_port = energon_port

        self.sock = BluetoothSocket(RFCOMM)
        self.sock.connect((device_address, 1))
        self.device_info = Info("energon_device_info", "Device info")
        self.device_info.info({"name": "UM25C", "device_address": self.device_address})

        self.volts = Gauge("energon_total_in_voltage_mV", "Volts")
        self.amps = Gauge("energon_total_in_current_A", "Amps")
        self.watts = Gauge("energon_total_in_power_mW", "Watts")
        self.temp_C = Gauge("energon_temp_C", "Temperature in C")
        self.temp_F = Gauge("energon_temp_F", "Temperature in F")
        self.group = Gauge("energon_group", "Measurement group")
        self.time = Gauge("energon_time", "Time")

    def __del__(self):
        self.sock.close()     

    def processdata(self, d):
        data = {}

        data["Volts"] = struct.unpack(">h", d[2: 3 + 1])[0] / 1000.0 * 1000.0   # millivolts
        data["Amps"] = struct.unpack(">h", d[4: 5 + 1])[0] / 10000.0            # amps
        data["Watts"] = struct.unpack(">I", d[6: 9 + 1])[0] / 1000.0 * 1000.0   # milliwatts
        data["temp_C"] = struct.unpack(">h", d[10: 11 + 1])[0]                  # temp in C
        data["temp_F"] = struct.unpack(">h", d[12: 13 + 1])[0]                  # temp in F
        data["group"] = struct.unpack(">h", d[14: 15 + 1])[0]                   # measurement group
        utc_dt = datetime.datetime.now(datetime.timezone.utc)                   # UTC time
        dt = utc_dt.astimezone()                                                # local time
        data["time"] = dt

        g = 0
        for i in range(16, 95, 8):
            ma, mw = struct.unpack(">II", d[i: i + 8])                  # mAh,mWh respectively
            gs = str(g)
            data[gs + "_mAh"] = ma
            data[gs + "_mWh"] = mw
            g += 1

        data["data_line_pos_volt"] = struct.unpack(">h", d[96: 97 + 1])[0] / 100.0
        data["data_line_neg_volt"] = struct.unpack(">h", d[98: 99 + 1])[0] / 100.0
        data["resistance"] = struct.unpack(">I", d[122: 125 + 1])[0] / 10.0             # resistance
        return data

    def query(self):
        d = b""
        while len(d) != 130:
            # Send request to USB meter
            self.sock.send((0xF0).to_bytes(1, byteorder="big"))
            d += self.sock.recv(130)
        data = self.processdata(d)
        self.volts.set(data["Volts"])
        self.amps.set(data["Amps"])
        self.watts.set(data["Watts"])
        self.temp_C.set(data["temp_C"])
        self.temp_F.set(data["temp_F"])
        self.group.set(data["group"])
        self.time.set(data["time"].timestamp())
        return data

    def run_metrics_loop(self):
        """Metrics fetching loop"""
        while True:
            print("------ new query ------")
            data = self.query()
            print(data)
            time.sleep(self.polling_interval_seconds)

    def run(self):
        """Run the metrics server"""
        print("Starting energon prometheus server")
        start_http_server(self.energon_port)
        print("Energon prometheus server running on port " + str(self.energon_port))
        self.run_metrics_loop()

# Example usage
if __name__ == "__main__":
    meter = UM25C("00:15:A3:00:55:02", 0.1, 8000)
    meter.run()