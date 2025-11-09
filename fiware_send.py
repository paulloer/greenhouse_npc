import requests

# Configuración del servidor FIWARE
FIWARE_URL = "http://iveg.ual.es:1026"
ENTITY_ID = "Agroconnect_Greenhouse_Reachers"

def fiware_send(value: float):
    # Cabeceras FIWARE
    headers = {
        "Content-Type": "application/json",
        "fiware-service": "Agroconnect",
        "fiware-servicepath": "/Greenhouse/Reachers"
    }

    # Datos a actualizar
    data = {
        "INVER_UVCEN11_REF": {"type": "Float", "value": value},
        "INVER_UVCEN12_REF": {"type": "Float", "value": value},
        "INVER_UVCEN13_REF": {"type": "Float", "value": value},
        "INVER_UVCEN21_REF": {"type": "Float", "value": value},
        "INVER_UVCEN22_REF": {"type": "Float", "value": value},
        "INVER_UVCEN23_REF": {"type": "Float", "value": value},
        "INVER_UVLAT1S_REF": {"type": "Float", "value": value},
        "INVER_UVLAT1SO_REF": {"type": "Float", "value": value},
        "INVER_UVLAT1NO_REF": {"type": "Float", "value": value},
        "INVER_UVLAT1N_REF": {"type": "Float", "value": value},
        "INVER_UVLAT2S_REF": {"type": "Float", "value": value},
        "INVER_UVLAT2E_REF": {"type": "Float", "value": value},
        "INVER_UVLAT2N_REF": {"type": "Float", "value": value}
    }

    # Endpoint para actualizar atributos
    url = f"{FIWARE_URL}/v2/entities/{ENTITY_ID}/attrs"

    # Realizar la petición PATCH
    try:
        response = requests.patch(url, headers=headers, json=data)
        
        if response.status_code == 204:
            print("Atributos actualizados correctamente")
        else:
            print(f"Error: {response.status_code}")
            print(f"Respuesta: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión: {e}")


if __name__ == '__main__':
    target_postion = 50
    fiware_send(target_postion)