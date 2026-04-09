use esp_idf_hal::gpio::{AnyIOPin, IOPin, PinDriver};
use esp_idf_hal::delay::Ets;

#[derive(Debug)]
pub struct DhtReading {
    pub temperature: f32,
    pub humidity: f32,
}

pub struct Dht11Sensor {
    pin: AnyIOPin,
}

impl Dht11Sensor {
    pub fn new(pin: impl IOPin + 'static) -> Self {
        Self {
            pin: pin.downgrade(),
        }
    }

    pub fn read(&mut self) -> Result<DhtReading, DhtError> {
        let mut data = [0u8; 5];

        // Send start signal: pull low 18ms, then high 40us
        {
            let mut pin = PinDriver::output(&mut self.pin).map_err(|_| DhtError::Gpio)?;
            pin.set_low().map_err(|_| DhtError::Gpio)?;
            Ets::delay_ms(18);
            pin.set_high().map_err(|_| DhtError::Gpio)?;
            Ets::delay_us(40);
        }

        // Switch to input and read response
        {
            let pin = PinDriver::input(&mut self.pin).map_err(|_| DhtError::Gpio)?;

            // Wait for sensor response low (80us)
            wait_for_level(&pin, false, 100)?;
            // Wait for sensor response high (80us)
            wait_for_level(&pin, true, 100)?;
            // Wait for first data bit low
            wait_for_level(&pin, false, 100)?;

            for byte in &mut data {
                for bit in (0..8).rev() {
                    // Wait for high (start of bit)
                    wait_for_level(&pin, true, 80)?;
                    // Measure high duration to determine 0 or 1
                    let high_us = measure_high_us(&pin, 100)?;
                    if high_us > 40 {
                        *byte |= 1 << bit;
                    }
                }
            }
        }

        let checksum = (data[0] as u16 + data[1] as u16 + data[2] as u16 + data[3] as u16) & 0xFF;
        if checksum != data[4] as u16 {
            return Err(DhtError::Checksum);
        }

        Ok(DhtReading {
            humidity: data[0] as f32 + data[1] as f32 * 0.1,
            temperature: data[2] as f32 + (data[3] & 0x7F) as f32 * 0.1,
        })
    }
}

fn wait_for_level(pin: &PinDriver<'_, AnyIOPin, esp_idf_hal::gpio::Input>, level: bool, timeout_us: u32) -> Result<(), DhtError> {
    for _ in 0..timeout_us {
        if pin.is_high() == level {
            return Ok(());
        }
        Ets::delay_us(1);
    }
    Err(DhtError::Timeout)
}

fn measure_high_us(pin: &PinDriver<'_, AnyIOPin, esp_idf_hal::gpio::Input>, timeout_us: u32) -> Result<u32, DhtError> {
    let mut count = 0;
    for _ in 0..timeout_us {
        if pin.is_low() {
            return Ok(count);
        }
        Ets::delay_us(1);
        count += 1;
    }
    Err(DhtError::Timeout)
}

#[derive(Debug)]
pub enum DhtError {
    Timeout,
    Checksum,
    Gpio,
}
