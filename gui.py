from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar, Style
import style
import threading

class Window(Frame):
	def __init__ (self, master = None):
		Frame.__init__(self, master)
		self.master = master
		self.style = Style()
		self.style.theme_use("clam")
		self.set_layout()
		self.init_stylizer()
		self.img_buffer = []
		self.img_buffer_index = -1
		self.run_count = 0
		self.working = False

	def stop_stylizer(self):
		if self.working:
			messagebox.showinfo("", "The thread will terminate soon. Please wait for the next message to confirm it before rerunning.")
			self.working = False
			self.stylizer.kill_yourself()

	def run_stylizer(self):
		if self.chosen_img_path is not None and self.style_img_path is not None:
			self.run_count = 0
			self.working = True
			self.stylizer.setup_run(self.chosen_img_path, self.style_img_path, self.result_save_path, self.update_progress, self.flag_finish)

	def update_progress(self, perc):
		print(f'{perc}% done')
		self.PROGRESS = perc
		self.PROGRESS_BAR['value'] = self.PROGRESS
		self.update_idletasks()

	def flag_finish(self, img):
		self.append_img_to_buffer(img)
		self.run_count += 1
		if self.run_count < 100 and self.working:
			return True
		else:
			self.working = False
			return False

	def pick_chosen_img(self):
		fn = filedialog.askopenfilename(initialdir="./images", filetypes = self.filetypes)
		if fn is not None:
			self.set_chosen_img(fn)
			self.chosen_img_path = fn

	def pick_style_img(self):
		fn = filedialog.askopenfilename(initialdir="./styles", filetypes = self.filetypes)
		if fn is not None:
			self.set_style_img(fn)
			self.style_img_path = fn

	def init_stylizer(self):
		self.stylizer = style.Stylizer(content_weight=400, style_weight=1600)
		self.chosen_img_path = None
		self.style_img_path = None
		self.result_save_path = "./partial_results"

	def clear_all(self):
		self.stop_stylizer()
		self.img_buffer = []
		self.img_buffer_index = -1
		self.set_chosen_img("./resources/placeholder_250x250.png")
		self.set_style_img("./resources/placeholder_250x250.png")
		self.set_current_img("./resources/placeholder_500x500.jpg")
		self.LABEL_COUNTER.configure(text="0/0")
		self.chosen_img_path = None
		self.style_img_path = None
		self.result_save_path = "./partial_results"
		self.init_stylizer()

	def set_layout(self):
		self.filetypes = [
			('images', '*.jpeg;*.jpg;*.bmp'),
			('all', '*')
		]
		self.configure(bg = "honeydew3")
		self.master.title("Style transfer app")
		self.pack(fill=BOTH, expand = 1)
		self.BUTTON_FRAME_TOP = ft = Frame(self, padx = 5, pady = 5, bg = "honeydew3")

		self.B_SELECT_CONTENT = Button(ft, text = "select content", command=self.pick_chosen_img, bg = "honeydew2")
		self.B_SELECT_STYLE = Button(ft, text = "select style", command = self.pick_style_img, bg = "honeydew2")
		self.B_RUN = Button(ft, text = "stylize", command = self.run_stylizer, bg = "honeydew2")
		self.CLEAR_ALL = Button(ft, text = "clear all", command = self.clear_all, bg = "honeydew2")
		self.B_TERMINATE = Button(ft, text = "stop processing", command = self.stop_stylizer, bg = "honeydew2")
		self.B_SAVE = Button(ft, text = "save currently visible", command = self.save_current, bg = "honeydew2")

		self.SECTION_CENTER = Frame(self, bg = "honeydew3")

		self.IMG_FRAME_LEFT = Frame(self.SECTION_CENTER, width = 210, bg = "honeydew3")
		self.IMG_CONTENT = Label(self.IMG_FRAME_LEFT, bg = "honeydew3")
		self.IMG_CONTENT_LABEL = Label(self.IMG_FRAME_LEFT, text="Content img", bg = "honeydew3")
		self.IMG_STYLE = Label(self.IMG_FRAME_LEFT, bg = "honeydew3")
		self.IMG_STYLE_LABEL = Label(self.IMG_FRAME_LEFT, text = "Style img", bg = "honeydew3")

		self.IMG_FRAME_CENTER = Frame(self.SECTION_CENTER, bg = "honeydew3")
		self.IMG_CURRENT = Label(self.IMG_FRAME_CENTER, bg = "honeydew3")
		self.LABEL_COUNTER = Label(self.IMG_FRAME_CENTER, text = "0/0", bg = "honeydew3")
		self.ARROW_BTNS_FRAME = Frame(self.IMG_FRAME_CENTER, bg = "honeydew3")
		self.ARROW_LEFT = Button(self.ARROW_BTNS_FRAME, text = "prev", command = self.pan_buff_left, bg = "honeydew2")
		self.PROGRESS = 0
		self.PROGRESS_BAR = Progressbar(self.ARROW_BTNS_FRAME, variable = self.PROGRESS, orient = HORIZONTAL, value = 0, maximum = 100, length = 100, mode = 'determinate')
		self.ARROW_RIGHT = Button(self.ARROW_BTNS_FRAME, text = "next", command = self.pan_buff_right, bg = "honeydew2")

		self.BUTTON_FRAME_TOP.pack(anchor = "w")
		self.B_SELECT_CONTENT.pack(side = LEFT, padx = 5, pady = 5)
		self.B_SELECT_STYLE.pack(side = LEFT, padx = 5, pady = 5)
		self.B_RUN.pack(side = LEFT, padx = 5, pady = 5)
		self.CLEAR_ALL.pack(side = LEFT, padx = 5, pady = 5)
		self.B_TERMINATE.pack(side = LEFT, padx = 5, pady = 5)
		self.B_SAVE.pack(side = LEFT, padx = 5, pady = 5)

		self.SECTION_CENTER.pack(anchor = "w")

		self.IMG_FRAME_LEFT.pack(side = LEFT)
		self.IMG_CONTENT.pack(side=TOP)
		self.IMG_CONTENT_LABEL.pack(side=TOP)
		self.IMG_STYLE.pack(side=TOP)
		self.IMG_STYLE_LABEL.pack(side=TOP)

		self.IMG_FRAME_CENTER.pack(anchor="w")
		self.IMG_CURRENT.pack(side = TOP)

		self.ARROW_BTNS_FRAME.pack(side = TOP)
		self.ARROW_LEFT.pack(side = LEFT)
		Frame(self.ARROW_BTNS_FRAME, bg = "honeydew3", width = 5).pack(side=LEFT)
		self.PROGRESS_BAR.pack(side = LEFT)
		Frame(self.ARROW_BTNS_FRAME, bg = "honeydew3", width = 5).pack(side=LEFT)
		self.ARROW_RIGHT.pack(side = LEFT)
		self.LABEL_COUNTER.pack(side = TOP)

		self.set_chosen_img("./resources/placeholder_250x250.png")
		self.set_style_img("./resources/placeholder_250x250.png")
		self.set_current_img("./resources/placeholder_500x500.jpg")

	def resize_by_ratio(self, img, cap):
		width, height = img.size
		ratio = width/height
		if ratio > 1:
			if width > cap:
				return img.resize((cap, int(cap/ratio)))
		elif height > cap:
			return img.resize((int(cap*ratio), cap))

		return img

	def set_chosen_img(self, path):
		img = ImageTk.PhotoImage(self.resize_by_ratio(Image.open(path), 250))
		self.IMG_CONTENT.img = img
		self.IMG_CONTENT.configure(image = img)

	def set_style_img(self, path):
		img = ImageTk.PhotoImage(self.resize_by_ratio(Image.open(path), 250))
		self.IMG_STYLE.img = img
		self.IMG_STYLE.configure(image=img)

	def set_current_img(self, path):
		_img = self.resize_by_ratio(Image.open(path), 500)
		
		img = ImageTk.PhotoImage(_img)
		self.IMG_CURRENT.img = img
		self.IMG_CURRENT.configure(image=img)

	def save_current(self):
		if len(self.img_buffer) < 1:
			messagebox.showerror("Buffer empty!")
			return
		fname = filedialog.asksaveasfilename(initialdir = './partial_results', defaultextension = '.jpg', filetypes = [('jpg','.jpg')])
		if fname is None:
			return
		self.img_buffer[self.img_buffer_index].save(fname)

	def pan_buff_right(self):
		if len(self.img_buffer) > 0:
			nindex = (self.img_buffer_index + 1) % len(self.img_buffer)
			self.set_current_img_by_index(nindex)
			self.LABEL_COUNTER.configure(text = f'{nindex + 1}/{len(self.img_buffer)}')

	def pan_buff_left(self):
		if len(self.img_buffer) > 0:
			move = self.img_buffer_index - 1
			if move < 0:
				move = len(self.img_buffer) - 1
			self.LABEL_COUNTER.configure(text = f'{move + 1}/{len(self.img_buffer)}')
			self.set_current_img_by_index(move)

	def append_img_to_buffer(self, img):
		self.img_buffer.append(img)
		self.set_current_img_by_index(len(self.img_buffer) - 1)
		self.LABEL_COUNTER.configure(text = f'{len(self.img_buffer)}/{len(self.img_buffer)}')

	def set_current_img_by_index(self, index):
		if index < 0 or index >= len(self.img_buffer):
			return
		self.img_buffer_index = index
		img = ImageTk.PhotoImage(self.img_buffer[index].resize((400,400)))
		self.IMG_CURRENT.img = img
		self.IMG_CURRENT.configure(image=img)

root = Tk()
root.configure(bg = "honeydew3")
root.geometry("800x600")
app = Window(root)

root.mainloop()
