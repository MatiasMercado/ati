import math
from kivy.graphics.context_instructions import Color
from kivy.graphics.vertex_instructions import Rectangle
from kivy.uix.boxlayout import BoxLayout


class Picture(BoxLayout):
    def __init__(self, img, is_color,  x_input, y_input, value_input, **kwargs):
        super(Picture, self).__init__(**kwargs)
        self.img = img
        self.is_color = is_color
        self.x_input = x_input
        self.y_input = y_input
        self.value_input = value_input
        self.origin_x = self.pos[0]
        self.origin_y = self.pos[1]
        self.rows = self.img.shape[0] + self.origin_x
        self.cols = self.img.shape[1] + self.origin_y
        self.selection_origin_coordinates = None
        self.selection_end_coordinates = None
        self.is_selected = False

    def on_touch_down(self, touch):
        if self.check_bounds(touch, self.rows, self.cols, self.origin_x, self.origin_y):
            print('You Just Touched: {}'.format(touch.pos))
            self.is_selected = False
            self.canvas.remove_group('selection')
            current_x = self.transform_x_coordinates(touch.x)
            current_y = self.transform_y_coordinates(touch.y)
            self.x_input.text = str(current_x)
            self.y_input.text = str(current_y)
            # Take coordinates (y,x) to use with the image matrix as (row, column)
            if self.is_color:
                r = int(self.img[current_y, current_x, 0])
                g = int(self.img[current_y, current_x, 1])
                b = int(self.img[current_y, current_x, 2])
                self.value_input.text = '{} {} {}'.format(r, g, b)
            else:
                self.value_input.text = str(int(self.img[current_y, current_x]))
            touch.ud['origin'] = (touch.x, touch.y)

    def transform_x_coordinates(self, x):
        return math.floor(x - self.origin_x)

    def transform_y_coordinates(self, y):
        return self.cols - math.floor(y) - 1

    def on_touch_move(self, touch):
        if self.check_bounds(touch, self.rows, self.cols, self.origin_x, self.origin_y) and 'origin' in touch.ud:
            selection_origin_x = touch.ud['origin'][0]
            selection_origin_y = touch.ud['origin'][1]
            self.is_selected = True
            self.selection_origin_coordinates = touch.ud['origin']
            self.selection_end_coordinates = touch.pos

            height = math.fabs(touch.y - selection_origin_y)
            base = math.fabs(touch.x - selection_origin_x )
            pos = self.calculate_rectangle_pos(selection_origin_x , selection_origin_y, touch.x, touch.y)
            self.canvas.remove_group('selection')
            with self.canvas:
                Color(1, 1, 1, 0.3)
                Rectangle(source='../resources/highlight', pos=(pos[0], pos[1]), size=(base, height),
                          group='selection')

    def check_bounds(self, touch, rows, cols, origin_x, origin_y):
        if origin_x <= touch.x < rows and origin_y <= touch.y < cols:
            return True
        return False

    def is_selected(self):
        return self.is_selected

    def get_selection_coordinates(self):
        origin_x = self.transform_x_coordinates(self.selection_origin_coordinates[0])
        origin_y = self.transform_y_coordinates(self.selection_origin_coordinates[1])
        end_x = self.transform_x_coordinates(self.selection_end_coordinates[0])
        end_y = self.transform_y_coordinates(self.selection_end_coordinates[1])
        # Take coordinates (y,x) to use with the image matrix as (row, column)
        origin = (origin_y, origin_x)
        end = (end_y, end_x)
        return [origin, end]

    def calculate_rectangle_pos(self, origin_x, origin_y, end_x, end_y):
        height = math.fabs(end_y - origin_y)
        base = math.fabs(end_x - origin_x)
        pos = (origin_x, origin_y)

        if end_x > origin_x:
            if end_y < origin_y:
                pos = (origin_x, origin_y - height)
        else:
            if end_y > origin_y:
                pos = (origin_x - base, origin_y)
            else:
                pos = (end_x, end_y)
        return pos


