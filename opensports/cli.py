"""
OpenSports CLI Tool

Comprehensive command-line interface for managing the OpenSports platform.
Includes commands for data ingestion, model training, system management, and more.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
from opensports.core.config import settings
from opensports.core.logging import get_logger, setup_logging
from opensports.core.database import get_database, initialize_database
from opensports.ingestion.collector import DataCollector
from opensports.modeling.player_performance import PlayerPerformanceModel
from opensports.modeling.game_predictor import GameOutcomePredictor
from opensports.segmentation.audience import AudienceSegmenter
from opensports.experiments.causal_analysis import CausalAnalyzer
from opensports.agents.game_analyst import GameAnalystAgent
from opensports.realtime.stream_processor import StreamProcessor

console = Console()
logger = get_logger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', help='Path to configuration file')
@click.pass_context
def cli(ctx, verbose, config):
    """
    🏀 OpenSports CLI - World-class sports analytics platform
    
    Author: Nik Jois <nikjois@llamaearch.ai>
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config
    
    # Setup logging
    setup_logging(level='DEBUG' if verbose else 'INFO')
    
    # Display banner
    if not ctx.invoked_subcommand:
        console.print(Panel.fit(
            "[bold blue]OpenSports CLI[/bold blue]\n"
            "[dim]World-class sports analytics platform[/dim]\n\n"
            "[green]Author:[/green] Nik Jois <nikjois@llamaearch.ai>\n"
            "[green]Version:[/green] 1.0.0",
            title="🏀 OpenSports",
            border_style="blue"
        ))


@cli.group()
def data():
    """Data management commands."""
    pass


@data.command()
@click.option('--sport', '-s', default='nba', help='Sport to collect data for')
@click.option('--days', '-d', default=7, help='Number of days to collect')
@click.option('--force', '-f', is_flag=True, help='Force re-collection of existing data')
def collect(sport, days, force):
    """Collect sports data from various APIs."""
    console.print(f"[blue]Collecting {sport.upper()} data for the last {days} days...[/blue]")
    
    async def run_collection():
        collector = DataCollector()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Collecting data...", total=None)
            
            try:
                result = await collector.collect_recent_data(
                    sport=sport,
                    days=days,
                    force_refresh=force
                )
                
                progress.update(task, description="Collection complete!")
                
                # Display results
                table = Table(title=f"{sport.upper()} Data Collection Results")
                table.add_column("Category", style="cyan")
                table.add_column("Count", style="green")
                table.add_column("Status", style="yellow")
                
                for category, data in result.items():
                    if isinstance(data, dict) and 'count' in data:
                        table.add_row(
                            category.replace('_', ' ').title(),
                            str(data['count']),
                            data.get('status', 'Success')
                        )
                
                console.print(table)
                
            except Exception as e:
                progress.update(task, description="Collection failed!")
                console.print(f"[red]Error: {str(e)}[/red]")
                sys.exit(1)
    
    asyncio.run(run_collection())


@data.command()
@click.option('--output', '-o', default='data_export.json', help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'parquet']), default='json')
@click.option('--sport', '-s', help='Filter by sport')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
def export(output, format, sport, start_date, end_date):
    """Export data from the database."""
    console.print(f"[blue]Exporting data to {output} in {format} format...[/blue]")
    
    async def run_export():
        db = get_database()
        
        # Build query filters
        filters = []
        if sport:
            filters.append(f"sport = '{sport}'")
        if start_date:
            filters.append(f"date >= '{start_date}'")
        if end_date:
            filters.append(f"date <= '{end_date}'")
        
        where_clause = " WHERE " + " AND ".join(filters) if filters else ""
        
        try:
            # Export games data
            query = f"SELECT * FROM games{where_clause}"
            games_data = await db.fetch_all(query)
            
            if format == 'json':
                export_data = {
                    'games': [dict(row) for row in games_data],
                    'exported_at': datetime.now().isoformat(),
                    'filters': {
                        'sport': sport,
                        'start_date': start_date,
                        'end_date': end_date
                    }
                }
                
                with open(output, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            elif format == 'csv':
                df = pd.DataFrame([dict(row) for row in games_data])
                df.to_csv(output, index=False)
            
            elif format == 'parquet':
                df = pd.DataFrame([dict(row) for row in games_data])
                df.to_parquet(output)
            
            console.print(f"[green]Exported {len(games_data)} records to {output}[/green]")
            
        except Exception as e:
            console.print(f"[red]Export failed: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run_export())


@cli.group()
def models():
    """Machine learning model commands."""
    pass


@models.command()
@click.option('--model', '-m', type=click.Choice(['player', 'game', 'all']), default='all')
@click.option('--sport', '-s', default='nba', help='Sport to train models for')
@click.option('--epochs', '-e', default=100, help='Number of training epochs')
def train(model, sport, epochs):
    """Train machine learning models."""
    console.print(f"[blue]Training {model} model(s) for {sport.upper()}...[/blue]")
    
    async def run_training():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            if model in ['player', 'all']:
                task = progress.add_task("Training player performance model...", total=None)
                try:
                    player_model = PlayerPerformanceModel()
                    result = await player_model.train_model(sport=sport, epochs=epochs)
                    progress.update(task, description="Player model trained!")
                    
                    console.print(f"[green]Player Model Results:[/green]")
                    console.print(f"  Accuracy: {result.get('accuracy', 0):.3f}")
                    console.print(f"  R² Score: {result.get('r2_score', 0):.3f}")
                    
                except Exception as e:
                    progress.update(task, description="Player model training failed!")
                    console.print(f"[red]Error: {str(e)}[/red]")
            
            if model in ['game', 'all']:
                task = progress.add_task("Training game prediction model...", total=None)
                try:
                    game_model = GameOutcomePredictor()
                    result = await game_model.train_model(sport=sport, epochs=epochs)
                    progress.update(task, description="Game model trained!")
                    
                    console.print(f"[green]Game Model Results:[/green]")
                    console.print(f"  Accuracy: {result.get('accuracy', 0):.3f}")
                    console.print(f"  Precision: {result.get('precision', 0):.3f}")
                    console.print(f"  Recall: {result.get('recall', 0):.3f}")
                    
                except Exception as e:
                    progress.update(task, description="Game model training failed!")
                    console.print(f"[red]Error: {str(e)}[/red]")
    
    asyncio.run(run_training())


@models.command()
@click.option('--model', '-m', required=True, help='Model name to evaluate')
@click.option('--sport', '-s', default='nba', help='Sport to evaluate')
@click.option('--test-size', default=0.2, help='Test set size (0.0-1.0)')
def evaluate(model, sport, test_size):
    """Evaluate trained models."""
    console.print(f"[blue]Evaluating {model} model for {sport.upper()}...[/blue]")
    
    async def run_evaluation():
        try:
            if model == 'player':
                player_model = PlayerPerformanceModel()
                results = await player_model.evaluate_model(sport=sport, test_size=test_size)
            elif model == 'game':
                game_model = GameOutcomePredictor()
                results = await game_model.evaluate_model(sport=sport, test_size=test_size)
            else:
                console.print(f"[red]Unknown model: {model}[/red]")
                return
            
            # Display evaluation results
            table = Table(title=f"{model.title()} Model Evaluation")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for metric, value in results.items():
                if isinstance(value, float):
                    table.add_row(metric.replace('_', ' ').title(), f"{value:.4f}")
                else:
                    table.add_row(metric.replace('_', ' ').title(), str(value))
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Evaluation failed: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run_evaluation())


@cli.group()
def analytics():
    """Analytics and insights commands."""
    pass


@analytics.command()
@click.option('--game-id', '-g', required=True, help='Game ID to analyze')
@click.option('--type', '-t', type=click.Choice(['comprehensive', 'tactical', 'statistical']), default='comprehensive')
def analyze_game(game_id, type):
    """Analyze a specific game using AI."""
    console.print(f"[blue]Analyzing game {game_id} with {type} analysis...[/blue]")
    
    async def run_analysis():
        try:
            analyst = GameAnalystAgent()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Running AI analysis...", total=None)
                
                result = await analyst.analyze_live_game(
                    game_id=game_id,
                    analysis_type=type
                )
                
                progress.update(task, description="Analysis complete!")
            
            # Display results
            console.print(Panel(
                result['ai_analysis'],
                title=f"🤖 AI Analysis - {game_id}",
                border_style="green"
            ))
            
            # Show key metrics
            if 'key_metrics' in result:
                table = Table(title="Key Metrics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="yellow")
                
                for metric, value in result['key_metrics'].items():
                    table.add_row(metric.replace('_', ' ').title(), str(value))
                
                console.print(table)
            
        except Exception as e:
            console.print(f"[red]Analysis failed: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run_analysis())


@analytics.command()
@click.option('--team', '-t', required=True, help='Team to analyze')
@click.option('--change-date', '-d', required=True, help='Date of coaching change (YYYY-MM-DD)')
@click.option('--metric', '-m', default='win_percentage', help='Outcome metric to analyze')
def causal_analysis(team, change_date, metric):
    """Perform causal analysis on coaching changes."""
    console.print(f"[blue]Analyzing causal impact of coaching change for {team}...[/blue]")
    
    async def run_causal_analysis():
        try:
            analyzer = CausalAnalyzer()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Running causal analysis...", total=None)
                
                result = await analyzer.estimate_coaching_impact(
                    team=team,
                    coach_change_date=change_date,
                    outcome_metric=metric
                )
                
                progress.update(task, description="Causal analysis complete!")
            
            # Display results
            console.print(Panel(
                f"Team: {result['team']}\n"
                f"Change Date: {result['coach_change_date']}\n"
                f"Metric: {result['outcome_metric']}\n\n"
                f"Pre-treatment mean: {result['descriptive_stats']['pre_treatment_mean']:.3f}\n"
                f"Post-treatment mean: {result['descriptive_stats']['post_treatment_mean']:.3f}\n"
                f"Naive difference: {result['descriptive_stats']['naive_difference']:.3f}",
                title="Causal Analysis Results",
                border_style="blue"
            ))
            
            # Show treatment effects
            if 'treatment_effects' in result:
                table = Table(title="Treatment Effects")
                table.add_column("Method", style="cyan")
                table.add_column("Effect", style="green")
                table.add_column("P-value", style="yellow")
                
                for method, effect_data in result['treatment_effects'].items():
                    table.add_row(
                        method.replace('_', ' ').title(),
                        f"{effect_data['effect']:.4f}",
                        f"{effect_data['p_value']:.4f}"
                    )
                
                console.print(table)
            
        except Exception as e:
            console.print(f"[red]Causal analysis failed: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run_causal_analysis())


@cli.group()
def system():
    """System management commands."""
    pass


@system.command()
def status():
    """Check system status and health."""
    console.print("[blue]Checking OpenSports system status...[/blue]")
    
    async def check_status():
        status_data = {
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Check database
        try:
            await initialize_database()
            db = get_database()
            await db.execute("SELECT 1")
            status_data['components']['database'] = {'status': 'healthy', 'message': 'Connected'}
        except Exception as e:
            status_data['components']['database'] = {'status': 'unhealthy', 'message': str(e)}
        
        # Check cache
        try:
            from opensports.core.cache import initialize_cache, get_cache
            await initialize_cache()
            cache = get_cache()
            await cache.ping()
            status_data['components']['cache'] = {'status': 'healthy', 'message': 'Connected'}
        except Exception as e:
            status_data['components']['cache'] = {'status': 'unhealthy', 'message': str(e)}
        
        # Display status
        table = Table(title="System Health Check")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Message", style="dim")
        
        for component, data in status_data['components'].items():
            status_style = "green" if data['status'] == 'healthy' else "red"
            status_icon = "HEALTHY" if data['status'] == 'healthy' else "UNHEALTHY"
            
            table.add_row(
                component.title(),
                f"[{status_style}]{status_icon} {data['status'].title()}[/{status_style}]",
                data['message']
            )
        
        console.print(table)
        
        # Overall status
        all_healthy = all(
            comp['status'] == 'healthy' 
            for comp in status_data['components'].values()
        )
        
        overall_status = "All systems operational" if all_healthy else "Some systems need attention"
        console.print(f"\n[bold]{overall_status}[/bold]")
    
    asyncio.run(check_status())


@system.command()
@click.option('--port', '-p', default=8000, help='Port to run the server on')
@click.option('--host', '-h', default='0.0.0.0', help='Host to bind the server to')
@click.option('--workers', '-w', default=1, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(port, host, workers, reload):
    """Start the OpenSports API server."""
    console.print(f"[blue]Starting OpenSports API server on {host}:{port}...[/blue]")
    
    try:
        import uvicorn
        from opensports.api.main import app
        
        uvicorn.run(
            "opensports.api.main:app",
            host=host,
            port=port,
            workers=workers if not reload else 1,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Server failed to start: {str(e)}[/red]")
        sys.exit(1)


@system.command()
def init():
    """Initialize the OpenSports system."""
    console.print("[blue]Initializing OpenSports system...[/blue]")
    
    async def run_init():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Initialize database
            task1 = progress.add_task("Initializing database...", total=None)
            try:
                await initialize_database()
                progress.update(task1, description="Database initialized")
            except Exception as e:
                progress.update(task1, description="Database initialization failed")
                console.print(f"[red]Database error: {str(e)}[/red]")
                return
            
            # Initialize cache
            task2 = progress.add_task("Initializing cache...", total=None)
            try:
                from opensports.core.cache import initialize_cache
                await initialize_cache()
                progress.update(task2, description="Cache initialized")
            except Exception as e:
                progress.update(task2, description="Cache initialization failed")
                console.print(f"[red]Cache error: {str(e)}[/red]")
                return
            
            # Create directories
            task3 = progress.add_task("Creating directories...", total=None)
            try:
                directories = ['data', 'models', 'logs', 'exports']
                for directory in directories:
                    Path(directory).mkdir(exist_ok=True)
                progress.update(task3, description="Directories created")
            except Exception as e:
                progress.update(task3, description="Directory creation failed")
                console.print(f"[red]Directory error: {str(e)}[/red]")
                return
        
        console.print("\n[green]OpenSports system initialized successfully![/green]")
        console.print("\n[dim]Next steps:[/dim]")
        console.print("  1. Run 'opensports data collect' to gather initial data")
        console.print("  2. Run 'opensports models train' to train ML models")
        console.print("  3. Run 'opensports system serve' to start the API server")
    
    asyncio.run(run_init())


@cli.command()
def version():
    """Show version information."""
    console.print(Panel.fit(
        "[bold blue]OpenSports v1.0.0[/bold blue]\n"
        "[dim]World-class sports analytics platform[/dim]\n\n"
        "[green]Author:[/green] Nik Jois\n"
        "[green]Email:[/green] nikjois@llamaearch.ai\n"
        "[green]License:[/green] MIT\n"
        "[green]Python:[/green] 3.11+",
        title="Version Info",
        border_style="blue"
    ))


if __name__ == '__main__':
    cli() 