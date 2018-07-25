import os

class detect_environment:
    LOCAL = 1
    COLABORATORY = 2
    AZURE = 3

    detected_environment = LOCAL


    def __init__(self):
        self.detected_environment = self.LOCAL
        try:
            os.environ['DATALAB_SETTINGS_OVERRIDES']
            self.detected_environment = self.COLABORATORY
        except:
            pass

        if self.detected_environment is self.LOCAL:
            try:
                os.environ['AZURE_NOTEBOOKS_HOST']
                self.detected_environment = self.AZURE
            except:
                pass

    def __str__(self):
        if self.detected_environment == self.COLABORATORY:
            return 'Google Colaboratory'
        elif self.detected_environment == self.AZURE:
            return 'Microsoft Azure Notebook'
        return 'Local'

