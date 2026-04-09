use esp_idf_svc::mqtt::client::{EspMqttClient, MqttClientConfiguration};
use log::info;

use crate::config;

pub fn connect() -> EspMqttClient<'static> {
    let broker_url = if config::MQTT_USER.is_empty() {
        format!("mqtt://{}:{}", config::MQTT_SERVER, config::MQTT_PORT)
    } else {
        format!(
            "mqtt://{}:{}@{}:{}",
            config::MQTT_USER, config::MQTT_PASSWORD, config::MQTT_SERVER, config::MQTT_PORT
        )
    };

    let mqtt_config = MqttClientConfiguration {
        client_id: Some(config::DEVICE_ID),
        ..Default::default()
    };

    info!("[mqtt] connecting to {}:{}", config::MQTT_SERVER, config::MQTT_PORT);

    let client = EspMqttClient::new_cb(
        &broker_url,
        &mqtt_config,
        |event| {
            log::debug!("[mqtt] event: {:?}", event.payload());
        },
    )
    .expect("failed to create MQTT client");

    info!("[mqtt] connected");
    client
}
