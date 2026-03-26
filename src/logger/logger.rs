use chrono;
use std::fs::{OpenOptions};
use std::io::Write;
use std::sync::Mutex;
use lazy_static::lazy_static;

pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

impl LogLevel {
    fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warning => "WARNING",
            LogLevel::Error => "ERROR",
        }
    }
}

pub struct LogMessage {
    pub level: LogLevel,
    pub timestamp: String,
    pub message: String,
}

lazy_static! {
    static ref LOG_FILE: Mutex<Option<std::fs::File>> = Mutex::new(None);
}

pub fn init_logger(log_path: &str) {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .expect("Could not open log file");
    let mut log_file = LOG_FILE.lock().unwrap();
    *log_file = Some(file);
}

pub fn log(level: LogLevel, message: &str) {
    let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let log_message = LogMessage {
        level,
        timestamp,
        message: message.to_string(),
    };
    let formatted = format!("[{}] [{}] {}\n", log_message.timestamp, log_message.level.as_str(), log_message.message);
    print!("{}", formatted);
    if let Some(ref mut file) = *LOG_FILE.lock().unwrap() {
        let _ = file.write_all(formatted.as_bytes());
    }
}


