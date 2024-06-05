from kivy.app import App
from kivy.core.text import LabelBase
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
import FlowPathTest

class MyApp(App):
    def build(self):
        # 创建垂直布局
        layout = BoxLayout(orientation='vertical')
        LabelBase.register(name="jt", fn_regular="myfont.ttf")
        # 创建显示文本的标签
        self.label = Label(text="请点击下方计算按钮开始计算", size_hint=(1, 0.8), font_name='jt')

        # 创建按钮
        button = Button(text='计算', size_hint=(1, 0.2), font_name='jt')
        button.bind(on_press=self.on_button_press)

        # 添加到布局
        layout.add_widget(self.label)
        layout.add_widget(button)

        return layout

    def on_button_press(self, instance):
        # 更新标签文本
        self.label.text = FlowPathTest.func()

        # 创建并显示弹出提示框
        popup = Popup(title='Popup Alert',
                      content=Label(text='计算完成', font_name='jt'),
                      size_hint=(None, None), size=(200, 200))
        popup.open()

        # 点击任意位置关闭提示框
        popup.bind(on_touch_down=popup.dismiss)


if __name__ == '__main__':
    MyApp().run()