from controller.app_controller import AppController

def main():
    """Application entry point"""
    app = AppController()
    
    # Add command line argument parsing
    import sys
    
    # Set debug mode if requested
    if "--debug" in sys.argv:
        app.debug = True
    
    # Check for runtime parameter
    runtime = 300  # Default 5 minutes
    for i, arg in enumerate(sys.argv):
        if arg == "--runtime" and i+1 < len(sys.argv):
            try:
                runtime = int(sys.argv[i+1])
                print(f"Setting maximum runtime to {runtime} seconds")
            except ValueError:
                print(f"Invalid runtime value: {sys.argv[i+1]}, using default (300s)")
    
    # Store the runtime setting
    app.max_runtime = runtime
    
    app.run()

if __name__ == "__main__":
    main()