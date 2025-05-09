def format_data(data):
    """
    Format data for display
    
    Args:
        data: The data to format
        
    Returns:
        Formatted data string
    """
    if isinstance(data, dict):
        return "\n".join([f"{k}: {v}" for k, v in data.items()])
    return str(data)