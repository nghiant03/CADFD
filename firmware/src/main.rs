mod config;
mod dht;
mod mqtt;
mod wifi;

use std::thread;
use std::time::Duration;

use esp_idf_svc::sntp::{EspSntp, SyncStatus};
use log::info;

fn main() {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    info!("[CESTA] Device {} starting", config::DEVICE_ID);

    let _peripherals = esp_idf_hal::peripherals::Peripherals::take().unwrap();
    let dht_pin = unsafe { esp_idf_hal::gpio::AnyIOPin::new(config::DHT_PIN) };
    let mut dht_sensor = dht::Dht11Sensor::new(dht_pin);

    let _wifi = wifi::connect();
    sync_time();

    let topic = format!("{}{}", config::MQTT_TOPIC_PREFIX, config::DEVICE_ID);
    let mut mqtt_client = mqtt::connect();

    loop {
        match dht_sensor.read() {
            Ok(reading) => {
                let payload = serde_json::json!({
                    "device_id": config::DEVICE_ID,
                    "timestamp": timestamp_epoch(),
                    "temperature": reading.temperature,
                    "humidity": reading.humidity,
                });
                let msg = serde_json::to_string(&payload).unwrap();

                match mqtt_client.publish(&topic, esp_idf_svc::mqtt::client::QoS::AtLeastOnce, false, msg.as_bytes()) {
                    Ok(_) => info!("[MQTT] {} → {}", topic, msg),
                    Err(e) => log::error!("[MQTT] Publish failed: {:?}", e),
                }
            }
            Err(e) => {
                log::warn!("[DHT] Read failed: {:?}, skipping", e);
            }
        }

        thread::sleep(Duration::from_secs(config::SEND_INTERVAL_SECS));
    }
}

fn sync_time() {
    info!("[NTP] Syncing time from {}", config::NTP_SERVER);
    let sntp = EspSntp::new_default().expect("Failed to create SNTP client");

    let mut attempts = 0;
    while sntp.get_sync_status() != SyncStatus::Completed && attempts < 20 {
        thread::sleep(Duration::from_millis(500));
        attempts += 1;
    }

    if sntp.get_sync_status() == SyncStatus::Completed {
        info!("[NTP] Time synced");
    } else {
        log::warn!("[NTP] Sync timed out, timestamps may use uptime");
    }

    std::mem::forget(sntp);
}

fn timestamp_epoch() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
