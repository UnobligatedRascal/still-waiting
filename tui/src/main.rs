//! still-waiting TUI client
//!
//! SSH-accessible terminal UI for managing training jobs on NOUGHT.
//! Connects to the orchestrator API over HTTP.
//!
//! UnobligatedRascal — Making old hardware sing.

use anyhow::Result;
use chrono::DateTime;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Span, Spans},
    widgets::{Block, Borders, Paragraph, Table, TableState, Row, Cell},
    Frame, Terminal,
};
use serde::{Deserialize, Serialize};
use std::io;
use uuid::Uuid;

// ===== API Types =====

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum JobStatus {
    Queued,
    Running,
    Checkpointed,
    Paused,
    Completed,
    Failed,
    SurgicallyEdited,
}

#[derive(Debug, Clone, Deserialize)]
struct TrainingJob {
    id: Uuid,
    model_ref: String,
    status: JobStatus,
    current_step: u64,
    target_steps: Option<u64>,
    checkpoints: Vec<CheckpointMeta>,
    created_at: DateTime<chrono::Utc>,
    error_message: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct CheckpointMeta {
    step: u64,
    path: String,
    metrics: std::collections::HashMap<String, f64>,
}

// ===== App State =====

struct App {
    orchestrator_url: String,
    jobs: Vec<TrainingJob>,
    selected_job: Option<usize>,
    table_state: TableState,
    refresh_interval: std::time::Duration,
    last_refresh: std::time::Instant,
    message: String,
    quit: bool,
}

impl App {
    fn new(url: &str) -> Self {
        Self {
            orchestrator_url: url.to_string(),
            jobs: vec![],
            selected_job: None,
            table_state: TableState::default().with_selected(Some(0)),
            refresh_interval: std::time::Duration::from_secs(3),
            last_refresh: std::time::Instant::now(),
            message: String::new(),
            quit: false,
        }
    }

    async fn refresh_jobs(&mut self) -> Result<()> {
        let resp = reqwest::get(format!("{}/v1/training/jobs", self.orchestrator_url))
            .await?
            .json::<serde_json::Value>()
            .await?;

        if let Some(data) = resp.get("data").and_then(|d| d.as_array()) {
            self.jobs = data
                .iter()
                .filter_map(|v| serde_json::from_value::<TrainingJob>(v.clone()).ok())
                .collect();

            // Reset selection if needed
            if !self.jobs.is_empty() {
                self.table_state.select(Some(0));
            }
        }

        Ok(())
    }

    fn status_style(status: &JobStatus) -> Style {
        match status {
            JobStatus::Running => Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            JobStatus::Queued => Style::default().fg(Color::Yellow),
            JobStatus::Paused => Style::default().fg(Color::Magenta),
            JobStatus::Completed => Style::default().fg(Color::Cyan),
            JobStatus::Failed => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            JobStatus::SurgicallyEdited => Style::default().fg(Color::Purple).add_modifier(Modifier::BOLD),
            JobStatus::Checkpointed => Style::default().fg(Color::Blue),
        }
    }
}

// ===== UI Rendering =====

fn render(f: &mut Frame<'_>, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints(
            [
                Constraint::Length(3), // Header
                Constraint::Min(10),  // Job table
                Constraint::Length(3), // Status bar
            ]
            .as_ref(),
        )
        .split(f.area());

    // Header
    let header = Paragraph::new(Spans::from(vec![
        Span::styled("still-waiting", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(" — "),
        Span::styled("NOUGHT Training Control", Style::default().fg(Color::Gray)),
        Span::raw(" ["),
        Span::styled("r", Style::default().fg(Color::Yellow).add_modifier(Modifier::UNDERLINED)),
        Span::raw("] refresh  ["),
        Span::styled("q", Style::default().fg(Color::Yellow).add_modifier(Modifier::UNDERLINED)),
        Span::raw("] quit"),
    ])))
    .block(Block::default().borders(Borders::ALL).title(" Jobs Overview "));
    f.render_widget(header, chunks[0]);

    // Job table
    let headers = vec![
        "ID",
        "Model",
        "Status",
        "Step",
        "Checkpoints",
        "Started",
    ];

    let rows = app.jobs.iter().map(|job| {
        let steps = match job.target_steps {
            Some(target) => format!("{}/{}", job.current_step, target),
            None => format!("{}", job.current_step),
        };
        let ckpts = job.checkpoints.len().to_string();
        let started = job.created_at.format("%H:%M:%S").to_string();

        Row::new(vec![
            Cell::from(job.id.to_string().get(0..8).unwrap_or(&job.id.to_string())),
            Cell::from(job.model_ref.split('/').last().unwrap_or(&job.model_ref)),
            Cell::from(format!("{}", job.status)).style(App::status_style(&job.status)),
            Cell::from(steps),
            Cell::from(ckpts),
            Cell::from(started),
        ])
    });

    let table = Table::new(
        rows,
        [
            Constraint::Length(10),
            Constraint::Min(20),
            Constraint::Length(16),
            Constraint::Length(12),
            Constraint::Length(10),
            Constraint::Length(10),
        ],
    )
    .header(Table::headers(vec![
        Cell::from("ID"),
        Cell::from("Model"),
        Cell::from("Status"),
        Cell::from("Step"),
        Cell::from("Checkpoints"),
        Cell::from("Started"),
    ]).style(Style::default().add_modifier(Modifier::BOLD)))
    .block(Block::default().borders(Borders::ALL).title(" Training Jobs "))
    .highlight_style(Style::default().bg(Color::DarkGray));

    f.render_stateful_widget(table, chunks[1], &mut app.table_state.clone());

    // Status bar
    let status_msg = if app.message.is_empty() {
        format!("Connected to {} | {} jobs | Last refresh: {}s ago",
            app.orchestrator_url,
            app.jobs.len(),
            app.last_refresh.elapsed().as_secs())
    } else {
        app.message.clone()
    };

    let status_bar = Paragraph::new(Spans::from(Span::raw(status_msg)))
        .style(Style::default().fg(Color::Gray))
        .block(Block::default().borders(Borders::ALL).title(" Status "));
    f.render_widget(status_bar, chunks[2]);
}

// ===== Main =====

#[tokio::main]
async fn main() -> Result<()> {
    let url = std::env::var("ORCH_URL").unwrap_or_else(|_| "http://localhost:8000".into());

    enable_raw_mode()?;
    let stdout = io::stdout();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    terminal.enter_alternate_screen()?;

    let mut app = App::new(&url);

    // Initial refresh
    if let Err(e) = app.refresh_jobs().await {
        app.message = format!("Error: {}", e);
    }
    app.last_refresh = std::time::Instant::now();

    loop {
        terminal.draw(|f| render(f, &app))?;

        if event::poll(std::time::Duration::from_millis(200))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => {
                            app.quit = true;
                        }
                        KeyCode::Char('r') => {
                            match app.refresh_jobs().await {
                                Ok(_) => {
                                    app.message = "Jobs refreshed".to_string();
                                    app.last_refresh = std::time::Instant::now();
                                }
                                Err(e) => {
                                    app.message = format!("Refresh failed: {}", e);
                                }
                            }
                        }
                        KeyCode::Up => {
                            let i = app.table_state.selected().unwrap_or(0);
                            if i > 0 {
                                app.table_state.select(Some(i - 1));
                            }
                        }
                        KeyCode::Down => {
                            let i = app.table_state.selected().unwrap_or(0);
                            if i + 1 < app.jobs.len() {
                                app.table_state.select(Some(i + 1));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Auto-refresh
        if app.last_refresh.elapsed() > app.refresh_interval {
            match app.refresh_jobs().await {
                Ok(_) => {
                    app.last_refresh = std::time::Instant::now();
                }
                Err(e) => {
                    app.message = format!("Auto-refresh failed: {}", e);
                }
            }
        }

        if app.quit {
            break;
        }
    }

    disable_raw_mode()?;
    terminal.leave_alternate_screen()?;

    Ok(())
}
