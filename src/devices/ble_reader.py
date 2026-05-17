"""BLE medical device reader using bleak — reads standard GATT health profiles."""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass

from src.config import BLEServices

# Standard GATT characteristic UUIDs
PLX_CONTINUOUS = "00002a5f-0000-1000-8000-00805f9b34fb"
PLX_SPOT_CHECK = "00002a5e-0000-1000-8000-00805f9b34fb"
BP_MEASUREMENT = "00002a35-0000-1000-8000-00805f9b34fb"
GLUCOSE_MEASUREMENT = "00002a18-0000-1000-8000-00805f9b34fb"
TEMP_MEASUREMENT = "00002a1c-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT = "00002a37-0000-1000-8000-00805f9b34fb"


@dataclass
class VitalReading:
    device_type: str
    values: dict
    raw_bytes: bytes | None = None


async def read_pulse_oximeter(address: str) -> VitalReading:
    """Read SpO2 and pulse rate from a BLE pulse oximeter."""
    from bleak import BleakClient

    async with BleakClient(address) as client:
        data = await client.read_gatt_char(PLX_SPOT_CHECK)
        # PLX Spot-check: flags(1) + SpO2(2) + PulseRate(2)
        spo2 = struct.unpack_from("<H", data, 1)[0]
        pulse = struct.unpack_from("<H", data, 3)[0]
        return VitalReading(
            device_type="pulse_oximeter",
            values={"spo2": spo2, "pulse_rate": pulse},
            raw_bytes=data,
        )


async def read_heart_rate(address: str) -> VitalReading:
    """Read heart rate from a BLE heart rate monitor."""
    from bleak import BleakClient

    async with BleakClient(address) as client:
        data = await client.read_gatt_char(HR_MEASUREMENT)
        flags = data[0]
        if flags & 0x01:  # 16-bit HR
            hr = struct.unpack_from("<H", data, 1)[0]
        else:  # 8-bit HR
            hr = data[1]
        return VitalReading(
            device_type="heart_rate",
            values={"heart_rate": hr},
            raw_bytes=data,
        )


async def read_blood_pressure(address: str) -> VitalReading:
    """Read blood pressure from a BLE BP monitor."""
    from bleak import BleakClient

    async with BleakClient(address) as client:
        data = await client.read_gatt_char(BP_MEASUREMENT)
        # BP Measurement: flags(1) + systolic(2) + diastolic(2) + MAP(2)
        systolic = struct.unpack_from("<H", data, 1)[0]
        diastolic = struct.unpack_from("<H", data, 3)[0]
        mean_ap = struct.unpack_from("<H", data, 5)[0]
        return VitalReading(
            device_type="blood_pressure",
            values={
                "systolic_bp": systolic,
                "diastolic_bp": diastolic,
                "mean_arterial_pressure": mean_ap,
            },
            raw_bytes=data,
        )


async def read_glucometer(address: str) -> VitalReading:
    """Read glucose from a BLE glucometer."""
    from bleak import BleakClient

    async with BleakClient(address) as client:
        data = await client.read_gatt_char(GLUCOSE_MEASUREMENT)
        # Simplified: flags(1) + sequence(2) + base_time(7) + concentration(2)
        glucose = struct.unpack_from("<H", data, 10)[0]
        return VitalReading(
            device_type="glucometer",
            values={"glucose": glucose},
            raw_bytes=data,
        )


async def scan_medical_devices(timeout: float = 10.0) -> list[dict]:
    """Scan for nearby BLE medical devices with known GATT health services."""
    from bleak import BleakScanner

    ble_services = BLEServices()
    medical_uuids = {
        ble_services.pulse_oximeter: "Pulse Oximeter",
        ble_services.blood_pressure: "Blood Pressure Monitor",
        ble_services.glucometer: "Glucometer",
        ble_services.thermometer: "Thermometer",
        ble_services.heart_rate: "Heart Rate Monitor",
    }

    devices = await BleakScanner.discover(timeout=timeout)
    medical_devices = []
    for device in devices:
        for adv_uuid in (device.metadata or {}).get("uuids", []):
            if adv_uuid.lower() in medical_uuids:
                medical_devices.append({
                    "address": device.address,
                    "name": device.name or "Unknown",
                    "type": medical_uuids[adv_uuid.lower()],
                    "service_uuid": adv_uuid,
                })
    return medical_devices
