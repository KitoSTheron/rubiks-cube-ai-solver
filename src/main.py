from controller.app_controller import AppController

def main():
    """Application entry point"""
    app = AppController()
    
    # Add command line argument parsing
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        # Set debug mode
        app.debug = True
    
    app.run()

if __name__ == "__main__":
    main()