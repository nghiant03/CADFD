use esp_idf_svc::wifi::{BlockingWifi, ClientConfiguration, Configuration, EspWifi};
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_hal::modem::Modem;
use log::info;

use crate::config;

pub fn connect() -> BlockingWifi<EspWifi<'static>> {
    let sys_loop = EspSystemEventLoop::take().unwrap();
    let nvs = EspDefaultNvsPartition::take().unwrap();

    let modem = unsafe { Modem::new() };
    let esp_wifi = EspWifi::new(modem, sys_loop.clone(), Some(nvs)).unwrap();
    let mut wifi = BlockingWifi::wrap(esp_wifi, sys_loop).unwrap();

    wifi.set_configuration(&Configuration::Client(ClientConfiguration {
        ssid: config::WIFI_SSID.try_into().unwrap(),
        password: config::WIFI_PASSWORD.try_into().unwrap(),
        ..Default::default()
    }))
    .unwrap();

    info!("[wifi] connecting to {}", config::WIFI_SSID);
    wifi.start().unwrap();
    wifi.connect().unwrap();
    wifi.wait_netif_up().unwrap();

    let ip_info = wifi.wifi().sta_netif().get_ip_info().unwrap();
    info!("[wifi] connected — IP {}", ip_info.ip);

    wifi
}
