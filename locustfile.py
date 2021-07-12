from locust import HttpUser, task, between


sample_code_and_constraints = {
    "bottom_code": [
        [416, 14, 355, 419, 106, 43, 419, 269], [78, 490, 283, 431, 294, 30, 266, 228], [492, 42, 53, 478, 366, 439, 266, 228], [161, 175, 478, 102, 111, 478, 478, 386], [154, 110, 229, 10, 298, 10, 359, 269], [145, 42, 86, 42, 42, 348, 364, 433], [229, 497, 4, 82, 473, 88, 260, 232], [145, 52, 349, 439, 439, 439, 325, 246], [65, 446, 283, 370, 230, 490, 364, 324], [159, 78, 229, 111, 175, 111, 102, 353], [154, 510, 365, 53, 294, 427, 450, 169], [242, 115, 251, 71, 155, 455, 30, 81], [266, 325, 364, 444, 325, 509, 441, 385], [403, 467, 61, 146, 129, 61, 430, 128], [333, 131, 115, 91, 183, 106, 238, 58], [298, 65, 456, 237, 497, 405, 478, 178], [467, 323, 419, 97, 103, 59, 233, 58], [146, 351, 290, 6, 421, 13, 300, 232], [69, 253, 111, 35, 295, 131, 72, 78], [103, 275, 179, 145, 61, 497, 233, 489], [24, 177, 163, 214, 338, 101, 352, 178], [387, 206, 313, 53, 238, 334, 118, 257], [175, 408, 13, 354, 188, 414, 300, 54], [49, 265, 198, 468, 249, 69, 158, 489], [106, 403, 371, 131, 497, 446, 102, 174], [121, 362, 33, 484, 195, 419, 361, 434], [167, 453, 333, 123, 157, 350, 175, 416], [138, 429, 507, 468, 412, 429, 361, 232], [33, 396, 492, 447, 472, 220, 175, 169], [333, 61, 333, 326, 333, 221, 403, 232], [108, 486, 486, 486, 417, 307, 307, 388], [367, 39, 447, 447, 59, 407, 294, 169], [413, 145, 367, 175, 145, 281, 263, 21], [49, 259, 221, 421, 150, 253, 186, 477], [332, 120, 403, 175, 77, 412, 206, 372], [454, 105, 129, 511, 350, 203, 145, 134], [262, 410, 285, 404, 141, 248, 182, 388], [415, 76, 231, 250, 428, 168, 124, 50], [131, 492, 414, 291, 59, 290, 373, 232], [455, 15, 332, 239, 475, 106, 42, 489], [405, 69, 115, 354, 287, 195, 15, 481], [495, 204, 488, 151, 306, 202, 341, 41], [446, 39, 131, 191, 14, 115, 431, 21], [389, 419, 193, 290, 229, 148, 370, 382], [58, 298, 473, 309, 110, 13, 497, 400], [322, 355, 29, 403, 65, 427, 497, 360], [82, 509, 204, 266, 136, 88, 24, 278], [181, 416, 378, 41, 29, 434, 195, 360], [309, 329, 204, 65, 24, 473, 289, 16], [197, 310, 193, 188, 230, 257, 183, 372], [400, 21, 197, 402, 181, 489, 311, 353], [342, 42, 501, 150, 444, 326, 110, 265], [489, 278, 200, 344, 169, 18, 174, 396], [309, 115, 33, 58, 358, 148, 211, 418], [305, 174, 200, 7, 416, 489, 120, 374], [305, 148, 378, 116, 13, 434, 420, 477], [148, 126, 148, 360, 195, 174, 89, 433], [232, 257, 148, 295, 420, 162, 138, 58], [297, 360, 372, 116, 399, 58, 399, 460], [265, 408, 390, 188, 390, 345, 390, 246], [390, 408, 408, 120, 388, 362, 311, 396], [278, 477, 388, 117, 481, 318, 174, 318], [62, 481, 41, 360, 7, 386, 324, 481], [174, 318, 120, 134, 50, 429, 386, 385]],
    "bottom_conditioning": {
        "instrument_family_str": [["organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ"]],
        "pitch": [[73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73]]
    },
    "mask": [[False, False, False, False], [True, True, True, True], [False, False, False, False], [False, False, False, False], [False, False, False, False], [True, True, True, True], [False, True, False, False], [False, True, False, False], [False, True, False, False], [False, True, False, False], [False, True, False, False], [False, True, False, False], [False, True, False, False], [False, True, False, False], [False, True, False, False], [False, True, False, False], [False, True, False, False], [False, True, False, False], [False, True, False, False], [False, True, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False]],
    "top_code": [[385, 257, 270, 489], [248, 449, 90, 505], [28, 393, 393, 488], [429, 307, 217, 440], [483, 310, 507, 197], [451, 445, 445, 21], [43, 348, 17, 207], [22, 376, 376, 469], [267, 415, 337, 323], [94, 169, 498, 165], [489, 70, 372, 382], [358, 483, 406, 425], [357, 358, 220, 86], [332, 273, 481, 86], [258, 142, 205, 242], [258, 147, 216, 425], [483, 91, 409, 165], [17, 440, 225, 86], [503, 90, 509, 44], [69, 165, 104, 165], [187, 324, 17, 50], [364, 17, 494, 265], [9, 90, 369, 26], [119, 329, 425, 9], [424, 246, 289, 453], [89, 1, 207, 319], [378, 241, 348, 198], [504, 482, 463, 66], [305, 207, 200, 207], [4, 362, 139, 69], [23, 207, 167, 498], [198, 483, 1, 1]],
    "top_conditioning": {
        "instrument_family_str": [["organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ", "organ"]],
        "pitch": [[73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73]]
    }
}


class NotonoUser(HttpUser):
    wait_time = between(1, 8)

    @task(0)
    def timerange_change(self):
        global sample_code_and_constraints
        self.client.post(
          "/timerange-change?"
          "&layer=top&start_index_top=0&temperature=0.8"
          "&pitch=47&instrument_family_str=mallet",
          json=sample_code_and_constraints)

    @task
    def get_spectrogram_image(self):
        global sample_code_and_constraints
        self.client.post(
          "/get-spectrogram-image",
          json=sample_code_and_constraints)

    @task(0)
    def get_audio(self):
        global sample_code_and_constraints
        self.client.post(
          "/get-audio",
          json=sample_code_and_constraints)
