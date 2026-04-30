"""
Credit Card Fraud Detection System - Main Entry Point
====================================================

This is the main entry point for running the complete fraud detection pipeline.

Usage:
    python main.py --help                    # Show help
    python main.py generate-data             # Generate synthetic dataset
    python main.py eda                       # Run exploratory data analysis
    python main.py train                     # Train all ML models
    python main.py predict                   # Run prediction demo
    python main.py api                       # Start FastAPI server
    python main.py all                       # Run complete pipeline

Examples:
    # Complete workflow
    python main.py all

    # Individual steps
    python main.py generate-data --samples 50000
    python main.py train --smote
    python main.py api --port 8000
"""

import argparse
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def banner():
    """Display welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           CREDIT CARD FRAUD DETECTION SYSTEM                      ║
║                                                                  ║
║  A comprehensive ML system for real-time fraud detection          ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def generate_data(args):
    """Generate synthetic transaction data."""
    from src.generate_synthetic_data import generate_dataset, save_dataset
    
    print("\n[STEP 1] Generating Synthetic Dataset")
    print("-" * 50)
    
    df = generate_dataset(n_samples=args.samples, fraud_ratio=args.fraud_ratio)
    save_dataset(df, output_dir=args.output_dir)
    
    print("\nDataset generation complete!")
    print(f"Location: {args.output_dir}/")
    return df


def run_eda(args):
    """Run exploratory data analysis."""
    from src.eda import generate_full_eda
    
    print("\n[STEP 2] Exploratory Data Analysis")
    print("-" * 50)
    
    data_path = args.data_path or 'data/transactions.parquet'
    stats = generate_full_eda(data_path, output_dir=args.output_dir)
    
    print("\nEDA complete!")
    print(f"Plots saved to: {args.output_dir}/")
    return stats


def train_models(args):
    """Train ML models."""
    from src.model_trainer import train_fraud_detection_model
    
    print("\n[STEP 3] Training ML Models")
    print("-" * 50)
    
    data_path = args.data_path or 'data/transactions.parquet'
    trainer = train_fraud_detection_model(
        data_path=data_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        use_smote=args.smote
    )
    
    print("\nTraining complete!")
    print(f"Model saved to: {args.model_dir}/")
    return trainer


def run_prediction(args):
    """Run prediction demo."""
    from src.predictor import demonstrate_prediction
    
    print("\n[STEP 4] Running Prediction Demo")
    print("-" * 50)
    
    try:
        demonstrate_prediction()
        print("\nPrediction demo complete!")
    except FileNotFoundError:
        print("\nError: Model not found. Please train the model first:")
        print("  python main.py train")


def start_api(args):
    """Start FastAPI server."""
    import uvicorn
    
    print("\n[STEP 5] Starting API Server")
    print("-" * 50)
    
    print(f"Starting server on http://localhost:{args.port}")
    print(f"API Documentation: http://localhost:{args.port}/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload
    )


def run_all(args):
    """Run complete pipeline."""
    banner()
    
    # Step 1: Generate data
    generate_data(args)
    
    # Step 2: EDA
    run_eda(args)
    
    # Step 3: Train models
    train_models(args)
    
    # Step 4: Prediction demo
    run_prediction(args)
    
    print("\n" + "=" * 60)
    print("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. View EDA plots in: outputs/")
    print("  2. Check model metrics in outputs/")
    print("  3. Start API server: python main.py api")
    print("  4. Launch dashboard: cd dashboard && npm run dev")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py generate-data --samples 100000
  python main.py train --smote
  python main.py api --port 8000
  python main.py all
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate data command
    gen_parser = subparsers.add_parser('generate-data', help='Generate synthetic dataset')
    gen_parser.add_argument('--samples', type=int, default=100000, help='Number of transactions')
    gen_parser.add_argument('--fraud-ratio', type=float, default=0.015, help='Fraud ratio')
    gen_parser.add_argument('--output-dir', default='data', help='Output directory')
    
    # EDA command
    eda_parser = subparsers.add_parser('eda', help='Run exploratory data analysis')
    eda_parser.add_argument('--data-path', default='data/transactions.parquet', help='Data file path')
    eda_parser.add_argument('--output-dir', default='outputs', help='Output directory')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument('--data-path', default='data/transactions.parquet', help='Data file path')
    train_parser.add_argument('--model-dir', default='models', help='Model directory')
    train_parser.add_argument('--output-dir', default='outputs', help='Output directory')
    train_parser.add_argument('--smote', action='store_true', help='Use SMOTE for balancing')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run prediction demo')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start API server')
    api_parser.add_argument('--port', type=int, default=8000, help='Server port')
    api_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    # All command (complete pipeline)
    all_parser = subparsers.add_parser('all', help='Run complete pipeline')
    all_parser.add_argument('--samples', type=int, default=50000, help='Number of transactions')
    all_parser.add_argument('--fraud-ratio', type=float, default=0.015, help='Fraud ratio')
    all_parser.add_argument('--smote', action='store_true', default=True, help='Use SMOTE')
    all_parser.add_argument('--data-dir', default='data', help='Data directory')
    all_parser.add_argument('--model-dir', default='models', help='Model directory')
    all_parser.add_argument('--output-dir', default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Run appropriate command
    commands = {
        'generate-data': generate_data,
        'eda': run_eda,
        'train': train_models,
        'predict': run_prediction,
        'api': start_api,
        'all': run_all
    }
    
    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
