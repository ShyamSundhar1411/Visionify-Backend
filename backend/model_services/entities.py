class ResponseMessage:
    def __init__(self, status, message, data=None):
        self.status = status
        self.message = message
        self.data = data

    def to_dict(self):
        base_dict = {"status": self.status, "message": self.message, "data": self.data}
        return base_dict


class ResponseBuilder:
    @staticmethod
    def build_response(data, message, status):
        return ResponseMessage(status, message, data).to_dict()
