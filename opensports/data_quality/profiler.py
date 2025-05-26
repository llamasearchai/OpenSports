"""
Advanced Data Profiling System

Comprehensive data profiling with statistical analysis, quality assessment,
and automated report generation for sports analytics data.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport as PandasProfileReport
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ColumnProfile:
    """Profile information for a single column"""
    name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    memory_usage: int
    
    # Numeric statistics
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # String statistics
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    
    # Categorical statistics
    most_frequent: Optional[str] = None
    most_frequent_count: Optional[int] = None
    
    # Data quality indicators
    outlier_count: Optional[int] = None
    outlier_percentage: Optional[float] = None
    pattern_violations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'dtype': self.dtype,
            'null_count': self.null_count,
            'null_percentage': self.null_percentage,
            'unique_count': self.unique_count,
            'unique_percentage': self.unique_percentage,
            'memory_usage': self.memory_usage,
            'mean': self.mean,
            'median': self.median,
            'std': self.std,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'q25': self.q25,
            'q75': self.q75,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'avg_length': self.avg_length,
            'most_frequent': self.most_frequent,
            'most_frequent_count': self.most_frequent_count,
            'outlier_count': self.outlier_count,
            'outlier_percentage': self.outlier_percentage,
            'pattern_violations': self.pattern_violations
        }


@dataclass
class DatasetProfile:
    """Profile information for entire dataset"""
    name: str
    shape: tuple
    memory_usage: int
    duplicate_rows: int
    duplicate_percentage: float
    missing_cells: int
    missing_percentage: float
    numeric_columns: int
    categorical_columns: int
    datetime_columns: int
    boolean_columns: int
    
    # Data quality scores
    completeness_score: float
    consistency_score: float
    validity_score: float
    overall_quality_score: float
    
    # Correlations
    high_correlations: List[tuple] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'shape': self.shape,
            'memory_usage': self.memory_usage,
            'duplicate_rows': self.duplicate_rows,
            'duplicate_percentage': self.duplicate_percentage,
            'missing_cells': self.missing_cells,
            'missing_percentage': self.missing_percentage,
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'datetime_columns': self.datetime_columns,
            'boolean_columns': self.boolean_columns,
            'completeness_score': self.completeness_score,
            'consistency_score': self.consistency_score,
            'validity_score': self.validity_score,
            'overall_quality_score': self.overall_quality_score,
            'high_correlations': self.high_correlations
        }


@dataclass
class ProfileReport:
    """Comprehensive profiling report"""
    dataset_profile: DatasetProfile
    column_profiles: List[ColumnProfile]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_time: float = 0.0
    
    # Visualizations
    correlation_matrix: Optional[str] = None
    distribution_plots: Dict[str, str] = field(default_factory=dict)
    quality_dashboard: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'dataset_profile': self.dataset_profile.to_dict(),
            'column_profiles': [cp.to_dict() for cp in self.column_profiles],
            'timestamp': self.timestamp.isoformat(),
            'execution_time': self.execution_time,
            'correlation_matrix': self.correlation_matrix,
            'distribution_plots': self.distribution_plots,
            'quality_dashboard': self.quality_dashboard
        }
    
    def export_html(self, filepath: str):
        """Export report as HTML"""
        html_content = self._generate_html_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_html_report(self) -> str:
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Profile Report - {self.dataset_profile.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9e9e9; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Profile Report</h1>
                <p><strong>Dataset:</strong> {self.dataset_profile.name}</p>
                <p><strong>Generated:</strong> {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Execution Time:</strong> {self.execution_time:.2f} seconds</p>
            </div>
            
            <div class="section">
                <h2>Dataset Overview</h2>
                <div class="metric">Rows: {self.dataset_profile.shape[0]:,}</div>
                <div class="metric">Columns: {self.dataset_profile.shape[1]:,}</div>
                <div class="metric">Memory Usage: {self.dataset_profile.memory_usage / 1024 / 1024:.2f} MB</div>
                <div class="metric">Overall Quality: {self.dataset_profile.overall_quality_score:.1%}</div>
            </div>
            
            <div class="section">
                <h2>Data Quality Metrics</h2>
                <div class="metric">Completeness: {self.dataset_profile.completeness_score:.1%}</div>
                <div class="metric">Consistency: {self.dataset_profile.consistency_score:.1%}</div>
                <div class="metric">Validity: {self.dataset_profile.validity_score:.1%}</div>
                <div class="metric">Duplicates: {self.dataset_profile.duplicate_percentage:.1%}</div>
            </div>
            
            <div class="section">
                <h2>Column Profiles</h2>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Null %</th>
                        <th>Unique %</th>
                        <th>Mean</th>
                        <th>Std</th>
                        <th>Quality Issues</th>
                    </tr>
        """
        
        for col in self.column_profiles:
            html += f"""
                    <tr>
                        <td>{col.name}</td>
                        <td>{col.dtype}</td>
                        <td>{col.null_percentage:.1%}</td>
                        <td>{col.unique_percentage:.1%}</td>
                        <td>{col.mean:.2f if col.mean is not None else 'N/A'}</td>
                        <td>{col.std:.2f if col.std is not None else 'N/A'}</td>
                        <td>{len(col.pattern_violations)} issues</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        return html


class DataProfiler:
    """Comprehensive data profiling system"""
    
    def __init__(self, enable_visualizations: bool = True):
        self.enable_visualizations = enable_visualizations
        self.plots_dir = "profile_plots"
        
    async def profile_dataset(self, data: pd.DataFrame, 
                            dataset_name: str = "dataset") -> ProfileReport:
        """Generate comprehensive dataset profile"""
        start_time = datetime.utcnow()
        
        # Generate dataset-level profile
        dataset_profile = await self._profile_dataset_overview(data, dataset_name)
        
        # Generate column-level profiles
        column_profiles = await self._profile_columns(data)
        
        # Create report
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        report = ProfileReport(
            dataset_profile=dataset_profile,
            column_profiles=column_profiles,
            execution_time=execution_time
        )
        
        # Generate visualizations if enabled
        if self.enable_visualizations:
            await self._generate_visualizations(data, report)
        
        return report
    
    async def _profile_dataset_overview(self, data: pd.DataFrame, 
                                      dataset_name: str) -> DatasetProfile:
        """Profile dataset overview"""
        # Basic statistics
        shape = data.shape
        memory_usage = data.memory_usage(deep=True).sum()
        duplicate_rows = data.duplicated().sum()
        duplicate_percentage = duplicate_rows / len(data) if len(data) > 0 else 0
        missing_cells = data.isnull().sum().sum()
        missing_percentage = missing_cells / (shape[0] * shape[1]) if shape[0] * shape[1] > 0 else 0
        
        # Column type counts
        numeric_columns = len(data.select_dtypes(include=[np.number]).columns)
        categorical_columns = len(data.select_dtypes(include=['object', 'category']).columns)
        datetime_columns = len(data.select_dtypes(include=['datetime64']).columns)
        boolean_columns = len(data.select_dtypes(include=['bool']).columns)
        
        # Quality scores
        completeness_score = 1 - missing_percentage
        consistency_score = self._calculate_consistency_score(data)
        validity_score = self._calculate_validity_score(data)
        overall_quality_score = (completeness_score + consistency_score + validity_score) / 3
        
        # High correlations
        high_correlations = self._find_high_correlations(data)
        
        return DatasetProfile(
            name=dataset_name,
            shape=shape,
            memory_usage=memory_usage,
            duplicate_rows=duplicate_rows,
            duplicate_percentage=duplicate_percentage,
            missing_cells=missing_cells,
            missing_percentage=missing_percentage,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            boolean_columns=boolean_columns,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            validity_score=validity_score,
            overall_quality_score=overall_quality_score,
            high_correlations=high_correlations
        )
    
    async def _profile_columns(self, data: pd.DataFrame) -> List[ColumnProfile]:
        """Profile individual columns"""
        column_profiles = []
        
        for column in data.columns:
            profile = await self._profile_single_column(data[column], column)
            column_profiles.append(profile)
        
        return column_profiles
    
    async def _profile_single_column(self, series: pd.Series, 
                                   column_name: str) -> ColumnProfile:
        """Profile a single column"""
        # Basic statistics
        null_count = series.isnull().sum()
        null_percentage = null_count / len(series) if len(series) > 0 else 0
        unique_count = series.nunique()
        unique_percentage = unique_count / len(series) if len(series) > 0 else 0
        memory_usage = series.memory_usage(deep=True)
        
        profile = ColumnProfile(
            name=column_name,
            dtype=str(series.dtype),
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            unique_percentage=unique_percentage,
            memory_usage=memory_usage
        )
        
        # Type-specific profiling
        if pd.api.types.is_numeric_dtype(series):
            await self._profile_numeric_column(series, profile)
        elif pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            await self._profile_string_column(series, profile)
        elif pd.api.types.is_categorical_dtype(series):
            await self._profile_categorical_column(series, profile)
        
        # Pattern violations and outliers
        profile.pattern_violations = self._detect_pattern_violations(series)
        if pd.api.types.is_numeric_dtype(series):
            outliers = self._detect_outliers(series)
            profile.outlier_count = len(outliers)
            profile.outlier_percentage = len(outliers) / len(series) if len(series) > 0 else 0
        
        return profile
    
    async def _profile_numeric_column(self, series: pd.Series, profile: ColumnProfile):
        """Profile numeric column"""
        clean_series = series.dropna()
        
        if len(clean_series) > 0:
            profile.mean = float(clean_series.mean())
            profile.median = float(clean_series.median())
            profile.std = float(clean_series.std())
            profile.min_value = float(clean_series.min())
            profile.max_value = float(clean_series.max())
            profile.q25 = float(clean_series.quantile(0.25))
            profile.q75 = float(clean_series.quantile(0.75))
            
            # Statistical measures
            try:
                profile.skewness = float(stats.skew(clean_series))
                profile.kurtosis = float(stats.kurtosis(clean_series))
            except:
                pass
    
    async def _profile_string_column(self, series: pd.Series, profile: ColumnProfile):
        """Profile string column"""
        clean_series = series.dropna().astype(str)
        
        if len(clean_series) > 0:
            lengths = clean_series.str.len()
            profile.min_length = int(lengths.min())
            profile.max_length = int(lengths.max())
            profile.avg_length = float(lengths.mean())
            
            # Most frequent value
            value_counts = clean_series.value_counts()
            if len(value_counts) > 0:
                profile.most_frequent = str(value_counts.index[0])
                profile.most_frequent_count = int(value_counts.iloc[0])
    
    async def _profile_categorical_column(self, series: pd.Series, profile: ColumnProfile):
        """Profile categorical column"""
        clean_series = series.dropna()
        
        if len(clean_series) > 0:
            value_counts = clean_series.value_counts()
            if len(value_counts) > 0:
                profile.most_frequent = str(value_counts.index[0])
                profile.most_frequent_count = int(value_counts.iloc[0])
    
    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """Calculate data consistency score"""
        consistency_issues = 0
        total_checks = 0
        
        for column in data.columns:
            total_checks += 1
            
            # Check for mixed data types in object columns
            if data[column].dtype == 'object':
                types = data[column].dropna().apply(type).unique()
                if len(types) > 1:
                    consistency_issues += 1
        
        return 1 - (consistency_issues / total_checks) if total_checks > 0 else 1.0
    
    def _calculate_validity_score(self, data: pd.DataFrame) -> float:
        """Calculate data validity score"""
        validity_issues = 0
        total_checks = 0
        
        for column in data.columns:
            total_checks += 1
            
            # Check for negative values in columns that shouldn't have them
            if column.lower() in ['age', 'score', 'points', 'goals', 'assists']:
                if pd.api.types.is_numeric_dtype(data[column]):
                    if (data[column] < 0).any():
                        validity_issues += 1
        
        return 1 - (validity_issues / total_checks) if total_checks > 0 else 1.0
    
    def _find_high_correlations(self, data: pd.DataFrame, threshold: float = 0.8) -> List[tuple]:
        """Find highly correlated column pairs"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return []
        
        corr_matrix = numeric_data.corr()
        high_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_correlations.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))
        
        return high_correlations
    
    def _detect_pattern_violations(self, series: pd.Series) -> List[str]:
        """Detect pattern violations in data"""
        violations = []
        
        # Email pattern for email columns
        if 'email' in series.name.lower():
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            invalid_emails = series.dropna().str.match(email_pattern) == False
            if invalid_emails.any():
                violations.append(f"Invalid email format: {invalid_emails.sum()} records")
        
        # Phone pattern for phone columns
        if 'phone' in series.name.lower():
            phone_pattern = r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
            invalid_phones = series.dropna().str.match(phone_pattern) == False
            if invalid_phones.any():
                violations.append(f"Invalid phone format: {invalid_phones.sum()} records")
        
        return violations
    
    def _detect_outliers(self, series: pd.Series, method: str = 'iqr') -> List[int]:
        """Detect outliers in numeric data"""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return []
        
        if method == 'iqr':
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
            return outliers.index.tolist()
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(clean_series))
            outliers = clean_series[z_scores > 3]
            return outliers.index.tolist()
        
        return []
    
    async def _generate_visualizations(self, data: pd.DataFrame, report: ProfileReport):
        """Generate visualization plots"""
        import os
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Correlation matrix
        if len(data.select_dtypes(include=[np.number]).columns) > 1:
            report.correlation_matrix = await self._create_correlation_plot(data)
        
        # Distribution plots for numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns[:10]:  # Limit to first 10 columns
            plot_path = await self._create_distribution_plot(data[column], column)
            report.distribution_plots[column] = plot_path
        
        # Quality dashboard
        report.quality_dashboard = await self._create_quality_dashboard(report)
    
    async def _create_correlation_plot(self, data: pd.DataFrame) -> str:
        """Create correlation matrix plot"""
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            xaxis_title="Features",
            yaxis_title="Features"
        )
        
        plot_path = f"{self.plots_dir}/correlation_matrix.html"
        fig.write_html(plot_path)
        return plot_path
    
    async def _create_distribution_plot(self, series: pd.Series, column_name: str) -> str:
        """Create distribution plot for a column"""
        clean_series = series.dropna()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Distribution', 'Box Plot'),
            vertical_spacing=0.1
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=clean_series, name='Distribution'),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=clean_series, name='Box Plot'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"Distribution Analysis: {column_name}",
            showlegend=False
        )
        
        plot_path = f"{self.plots_dir}/distribution_{column_name}.html"
        fig.write_html(plot_path)
        return plot_path
    
    async def _create_quality_dashboard(self, report: ProfileReport) -> str:
        """Create data quality dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Quality Scores',
                'Column Types',
                'Missing Data by Column',
                'Data Quality Issues'
            ),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Quality scores
        quality_metrics = ['Completeness', 'Consistency', 'Validity', 'Overall']
        quality_scores = [
            report.dataset_profile.completeness_score,
            report.dataset_profile.consistency_score,
            report.dataset_profile.validity_score,
            report.dataset_profile.overall_quality_score
        ]
        
        fig.add_trace(
            go.Bar(x=quality_metrics, y=quality_scores, name='Quality Scores'),
            row=1, col=1
        )
        
        # Column types
        type_labels = ['Numeric', 'Categorical', 'DateTime', 'Boolean']
        type_values = [
            report.dataset_profile.numeric_columns,
            report.dataset_profile.categorical_columns,
            report.dataset_profile.datetime_columns,
            report.dataset_profile.boolean_columns
        ]
        
        fig.add_trace(
            go.Pie(labels=type_labels, values=type_values, name='Column Types'),
            row=1, col=2
        )
        
        # Missing data by column
        column_names = [cp.name for cp in report.column_profiles[:10]]
        missing_percentages = [cp.null_percentage for cp in report.column_profiles[:10]]
        
        fig.add_trace(
            go.Bar(x=column_names, y=missing_percentages, name='Missing Data'),
            row=2, col=1
        )
        
        # Data quality issues
        issue_counts = [len(cp.pattern_violations) for cp in report.column_profiles[:10]]
        
        fig.add_trace(
            go.Bar(x=column_names, y=issue_counts, name='Quality Issues'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Data Quality Dashboard",
            showlegend=False
        )
        
        plot_path = f"{self.plots_dir}/quality_dashboard.html"
        fig.write_html(plot_path)
        return plot_path
    
    def generate_pandas_profile(self, data: pd.DataFrame, 
                              title: str = "Data Profile Report") -> str:
        """Generate pandas profiling report"""
        try:
            profile = PandasProfileReport(
                data,
                title=title,
                explorative=True,
                minimal=False
            )
            
            output_path = f"{self.plots_dir}/pandas_profile_report.html"
            profile.to_file(output_path)
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating pandas profile: {e}")
            return "" 