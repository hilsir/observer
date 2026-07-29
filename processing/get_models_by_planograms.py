
from processing.recognition.model_selection import ModelManager
model_manager = ModelManager()

def get_models_by_planograms(planogram_names):
    selected_model = None

    for planogram_name in planogram_names:
        selected_model = model_manager.get_model_for(planogram_name)
        if selected_model is not None:
            break

    return selected_model