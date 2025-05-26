# OpenInsight Experiment Service Examples

This directory contains examples showing how to use the OpenInsight Experiment Service in different contexts.

## Web Integration Example

The `web_integration.py` example demonstrates how to integrate the experiment service with a web application using FastAPI. It shows:

1. How to create and manage experiments
2. How to assign variants to users based on cookies
3. How to track conversions from user interactions
4. How to analyze and display experiment results

### Running the Example

1. First, install the required dependencies:

```bash
pip install fastapi uvicorn jinja2
```

2. Run the example:

```bash
cd OpenInsight/experiments/examples
uvicorn web_integration:app --reload
```

3. Open your browser at [http://localhost:8000](http://localhost:8000)

4. To see the admin view with experiment results, visit [http://localhost:8000/?admin=true](http://localhost:8000/?admin=true)

### Simulating Traffic

To simulate traffic and generate results:

1. Open the application in multiple browser windows or use different browsers to get different user IDs
2. Click on the "Sign Up Now" and "Purchase Now" buttons to generate conversions
3. View the results in the admin view

### Key Concepts Demonstrated

- **User Identification**: Using cookies to maintain consistent user IDs
- **Variant Assignment**: Assigning variants to users based on their ID
- **Conversion Tracking**: Recording conversions when users take actions
- **Results Analysis**: Displaying experiment results with statistical analysis
- **Multi-Experiment Setup**: Running multiple experiments simultaneously

## SQL Persistence Example

The `sql_persistence_example.py` example shows how to use the SQL database backend for storing experiment data. It demonstrates:

1. Setting up a SQL database for experiments
2. Creating and running experiments with automatic persistence
3. Restarting the application and retrieving preserved experiment data
4. Continuing to collect data across application restarts

### Running the Example

1. First, install the required dependencies:

```bash
pip install sqlalchemy
```

2. To create a new experiment:

```bash
python sql_persistence_example.py --create
```

3. To simulate more traffic for an existing experiment:

```bash
python sql_persistence_example.py --users 200
```

### Key Concepts Demonstrated

- **Database Storage**: Persisting experiment data in a SQL database
- **Application Restart**: Maintaining experiment state across application restarts
- **Incremental Data Collection**: Adding more data to existing experiments
- **Sample Size Calculation**: Dynamically calculating required sample sizes
- **Winner Detection**: Automated detection of statistically significant winners

## Redis Persistence Example

The `redis_persistence_example.py` example demonstrates using Redis for high-performance storage of experiment data. It shows:

1. Setting up a Redis server for experiment storage
2. Running experiments with automatic Redis persistence
3. Handling high-traffic scenarios with multi-threading
4. Benchmarking performance of experiment operations

### Running the Example

1. First, install the required dependencies:

```bash
pip install redis
```

2. Start a Redis server (or use Docker):

```bash
# Using Docker
docker run --name redis-server -p 6379:6379 -d redis
```

3. To create a new experiment and run a high-volume simulation:

```bash
python redis_persistence_example.py --create --users 5000 --threads 8
```

4. To benchmark performance with an existing experiment:

```bash
python redis_persistence_example.py --benchmark
```

### Key Concepts Demonstrated

- **High-Performance Storage**: Using Redis for fast read/write operations
- **Concurrent Access**: Multi-threaded access to experiment data
- **Performance Benchmarking**: Measuring throughput of experiment operations
- **Automatic Persistence**: Redis-backed automatic saving of experiment data
- **Distributed Operations**: Support for distributed application scenarios

## Adding Your Own Examples

Feel free to add your own examples to this directory. Some ideas for additional examples:

- Mobile app integration
- Server-side rendering integration
- Integration with analytics platforms
- Advanced segmentation examples
- Custom statistical analysis 