# OpenInsight Experiment Service

The experiment service provides tools for running A/B tests and other types of experiments within the OpenInsight platform. It supports multiple experiment types, deterministic user assignment, and statistical analysis.

## Features

- **Multiple Experiment Types**
  - A/B Tests (split testing)
  - Multi-armed bandits (adaptive allocation)
  - Interleaving experiments

- **User Assignment**
  - Deterministic user assignment (users always get the same variant)
  - Traffic allocation (percentage of users included in experiment)

- **Comprehensive Analysis**
  - Statistical significance testing (Fisher's exact test)
  - Automatic winner determination
  - Relative improvement calculations

- **Conversion Tracking**
  - Binary conversions (success/failure)
  - Value-based conversions (revenue, engagement metrics)

- **Persistence Options**
  - JSON file storage (default)
  - SQL database storage via SQLAlchemy
  - Redis storage for high-performance applications
  - Auto-saving manager with configurable intervals

## Usage

### Creating an Experiment

```python
from OpenInsight.experiments import get_experiment_manager

# Get the experiment manager
manager = get_experiment_manager()

# Create a simple A/B test
experiment = manager.create_experiment(
    name="Homepage Button Color Test",
    variants=[
        {"name": "Blue Button", "description": "Standard blue button"},
        {"name": "Green Button", "description": "New green button"}
    ],
    experiment_type="ab_test",
    traffic_allocation=1.0,
    description="Testing whether a green CTA button performs better than our standard blue"
)
```

### Assigning Variants to Users

```python
# Assign a variant to a user
user_id = "user123"
variant = manager.get_variant_for_user(experiment.experiment_id, user_id)

if variant:
    # User is in the experiment
    button_color = "green" if variant["name"] == "Green Button" else "blue"
    # ... display appropriate button ...
else:
    # User is not in the experiment (due to traffic allocation)
    # ... display default button ...
```

### Recording Conversions

```python
# When a user converts (e.g., makes a purchase, signs up, etc.)
manager.record_conversion(
    experiment_id=experiment.experiment_id,
    variant_id=variant["variant_id"],
    value=purchase_amount  # Optional value of the conversion
)
```

### Analyzing Results

```python
# Get experiment analysis
analysis = manager.analyze_experiment(experiment.experiment_id)

# Check if there's a winner
if analysis["winner"]:
    winner = analysis["winner"]
    print(f"The winner is: {winner['name']}")
    print(f"Conversion rate: {winner['conversion_rate']:.2%}")
    print(f"p-value: {winner['p_value']:.4f}")
    print(f"Relative improvement: {winner['relative_improvement']:.2f}%")
else:
    print("No statistically significant winner yet.")
```

### Ending an Experiment

```python
# End the experiment
manager.end_experiment(experiment.experiment_id)
```

### Using SQL Database Persistence

```python
from OpenInsight.experiments import get_sql_persistent_manager

# Get a manager with SQL persistence
manager = get_sql_persistent_manager(
    connection_string="sqlite:///path/to/experiments.db",
    autosave_interval=60  # Save every 60 seconds
)

# Use the manager normally - all operations will be persisted to the database
experiment = manager.create_experiment(
    name="Persistent Experiment",
    variants=[
        {"name": "Control", "description": "Original version"},
        {"name": "Variant A", "description": "New version"}
    ],
    experiment_type="ab_test"
)

# The experiment data will be automatically saved to the database
# and will persist across application restarts
```

### Using Redis for High-Performance Applications

```python
from OpenInsight.experiments import get_redis_persistent_manager

# Get a manager with Redis persistence
manager = get_redis_persistent_manager(
    redis_url="redis://hostname:6379/0",  # Redis server address
    key_prefix="myapp:experiments:",  # Prefix for Redis keys
    autosave_interval=5,  # Save every 5 seconds for real-time syncing
    expire_time=604800  # Optional: expire keys after 1 week (in seconds)
)

# Create and use experiments as normal
experiment = manager.create_experiment(
    name="High Traffic Experiment",
    variants=[
        {"name": "Version A", "description": "Current version"},
        {"name": "Version B", "description": "New version"}
    ],
    experiment_type="ab_test"
)

# Redis provides high throughput for concurrent applications
# and supports distributed access from multiple application instances
```

## API Endpoints

The experiment service provides RESTful API endpoints:

- `POST /experiments/`: Create a new experiment
- `GET /experiments/`: List all experiments
- `GET /experiments/{experiment_id}`: Get experiment details
- `POST /experiments/{experiment_id}/assign`: Assign a variant to a user
- `POST /experiments/{experiment_id}/convert`: Record a conversion
- `GET /experiments/{experiment_id}/analyze`: Analyze experiment results
- `POST /experiments/{experiment_id}/end`: End an experiment
- `DELETE /experiments/{experiment_id}`: Delete an experiment

## Running the Demo

You can run the included demo script to see the experiment service in action:

```bash
python -m OpenInsight.experiments.demo
```

The demo shows:
1. A simple A/B test for button colors
2. A multi-armed bandit experiment for pricing optimization
3. Analysis and winner determination

## Implementation Details

- The experiment service uses the SciPy library for statistical analysis.
- Multi-armed bandit implementation uses Thompson sampling for adaptive allocation.
- Deterministic user assignment ensures consistent user experience.
- Fisher's exact test is used for statistical significance testing.
- Multiple storage backends support different persistence needs:
  - JSON file storage for simple deployments
  - SQL database storage for production environments
  - Redis storage for high-performance and distributed applications
  - Memory-only storage for testing

## Choosing a Storage Backend

- **JSON File Backend**: Simple and easy to set up, suitable for development and small applications
- **SQL Database Backend**: Reliable and robust, ideal for production environments with regular traffic
- **Redis Backend**: High-performance option for applications with heavy traffic or distributed architecture

## Best Practices

1. **Always define a control variant**: The first variant in your experiment is treated as the control.
2. **Run experiments long enough**: Statistical significance requires sufficient data.
3. **Use meaningful experiment names**: Make it easy to understand the purpose of each experiment.
4. **End experiments after making decisions**: Don't leave experiments running indefinitely.
5. **Consider segment analysis**: Different user segments may respond differently to variants.
6. **Use database persistence in production**: The SQL backend provides reliability for production environments.
7. **Calculate required sample sizes**: Use the sample size calculator to plan experiment duration.
8. **For high traffic applications, use Redis**: The Redis backend offers superior performance for high-volume scenarios. 