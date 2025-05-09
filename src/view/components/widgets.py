import tkinter as tk

class CustomFrame(tk.Frame):
    """
    A custom frame widget that can be reused throughout the application.
    """
    
    def __init__(self, parent, *args, **kwargs):
        """Initialize the custom frame"""
        super().__init__(parent, *args, **kwargs)