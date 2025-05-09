from model.data_model import DataModel
from view.main_window import MainWindow

class AppController:
    """
    Application controller.
    Connects the model and view components.
    """
    
    def __init__(self):
        """Initialize the application controller"""
        self.model = DataModel()
        self.view = MainWindow(self)
    
    def run(self):
        """Start the application"""
        self.view.run()
    
    def update_model(self, key, value):
        """
        Update the model with new data
        
        Args:
            key: The data key
            value: The data value
        """
        self.model.set_data(key, value)
        # Update the view with the new data
        self.view.update_view(self.model.get_data())