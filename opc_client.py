from opcua import Client, ua
from time import time
from constants import URL, OPC_PARAMETERS, OPC_CONTROL

class gh_client:
    """
    A class to interact with an OPC UA server using the opcua library.
    """ 
    def __init__(self, url):
        """
        Initialize the OPC UA client with the given server URL.
        
        :param url: The URL of the OPC UA server (e.g., "opc.tcp://<server_ip>:<port>")
        """
        self.url = url
        self.client = Client(url)
        self.connected = False

    def read(self, variables: list, names=None):
        try:
            self.client.connect()
            # print("Connected to OPC UA Server")

            if names != None:
                keys = names
            else:
                keys = variables

            read_results = {}
            for idx, variable in enumerate(variables):
                node = self.client.get_node(f"ns=2;s=SCADA.ASP_NUEVO.{variable}.PresentValue")
                value = node.get_value()
                read_results[keys[idx]] = value

        finally:
            self.client.disconnect()

        return read_results

    def write(self, variables: list, value: float):
        """
        Write a value to a specified variable on the OPC UA server.
        
        :param variable: Name of the variable suffix (e.g., "TEMP_INTERIOR_S1")
        :param value: Value to write
        """
        try:
            self.client.connect()
            # print("Connected to OPC UA Server")

            for variable in variables:
                node = self.client.get_node(f"ns=2;s=SCADA.ASP_NUEVO.{variable}.PresentValue")
                variant = ua.Variant(value, ua.VariantType.Float)
                datavalue = ua.DataValue(variant)
                datavalue.ServerTimestamp = None
                datavalue.SourceTimestamp = None
                node.set_value(datavalue)
                # print(f"Wrote value {value} to variable {variable}")

        finally:
            self.client.disconnect()


if __name__ == "__main__":

    gh13_client = gh_client(URL)
    tic = time()
    read_results = gh13_client.read(OPC_PARAMETERS)
    toc = time()
    print(f"Time to read: {(toc - tic):.3f} seconds"), 
    print("Read values:", read_results)



    gh13_client.read('OPC_UVCEN1_1_POS_VALOR')
    gh13_client.write('OPC_UVCEN1_1_POS', 100.0)


    