# Firmware

## Overview

Rust firmware for ESP32-S3 with a DHT11 sensor, built with `esp-idf-hal` in the `std` environment.

## Requirements

Required tools:

- `espup` for the Espressif Rust toolchain
- `espflash` for flashing and serial monitoring
- `ldproxy` as required by `esp` toolchain
- ``
- ESP-IDF host dependencies required by the managed toolchain on your system

## Project Structure

```text
firmware/
├── Cargo.toml            # Dependencies: esp-idf-svc, esp-idf-hal, serde_json
├── build.rs              # ESP-IDF build integration via embuild
├── rust-toolchain.toml   # Uses the `esp` Rust toolchain in this directory
├── sdkconfig.defaults    # ESP-IDF Kconfig defaults (WiFi, SNTP, MQTT)
├── .cargo/config.toml    # Target: xtensa-esp32s3-espidf
└── src/
    ├── main.rs           # Entry point: init → WiFi → NTP → MQTT loop
    ├── config.rs         # WiFi SSID/password, MQTT broker, device ID, DHT pin
    ├── wifi.rs           # BlockingWifi connection via esp-idf-svc
    ├── mqtt.rs           # EspMqttClient connection and publish
    └── dht.rs            # Bit-banged DHT11 protocol over GPIO
```

## Configuration

This directory includes `rust-toolchain.toml`, so Cargo automatically selects the `esp` toolchain when run from `firmware/`.

## Usage

```bash
cd firmware
cargo check
cargo build --release
espflash flash target/xtensa-esp32s3-espidf/release/cesta-firmware --monitor
```

## MQTT Payload

Publishes JSON to `cesta/readings/<device_id>` every 30 seconds:

```json
{"device_id": "esp32_01", "timestamp": 1718000000, "temperature": 25.3, "humidity": 60.1}
```

## Deployment

ESP32 devices connect via WiFi to an on-prem MQTT broker. Recommended stack:

- **Mosquitto** — MQTT broker
- **Telegraf** — MQTT → InfluxDB bridge
- **InfluxDB** — Time-series storage
- **Grafana** — Dashboard
- **Python MQTT subscriber** — export to `data/raw/esp32_dht11/` CSV for the CESTA pipeline
