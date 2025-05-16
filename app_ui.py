import tkinter as tk
from tkinter import messagebox
from utils.input_handler import save_entry
from utils.viewer import view_summary
from utils.plotter import plot_readings
from utils.detector import detect_anomalies
from utils.model import train_lstm_model
from utils.reporter import generate_report
from datetime import datetime

LANGUAGES = {
    "en": {
        "title": "BPCompanion",
        "subtitle": "A personal BP monitor for Grandpa",
        "add_entry": "Add New BP Entry",
        "view_summary": "View Summary (Terminal)",
        "plot_trend": "Plot Trend (Matplotlib)",
        "detect_anomalies": "Detect Anomalies",
        "forecast": "LSTM Forecast",
        "report": "Generate Full Report (PDF)",
        "exit": "Exit",
        "form_title": "Enter Grandpa's BP Reading",
        "form_labels": ["Date (YYYY-MM-DD)", "Time (HH:MM)", "Systolic", "Diastolic", "Pulse", "Notes"],
        "form_button": "Save Entry",
        "saved": "Entry saved successfully!",
        "error_numbers": "Systolic, Diastolic, and Pulse must be integers.",
        "report_done": "Report saved as bp_full_report.pdf",
        "lang_toggle": "中文"
    },
    "zh": {
        "title": "血压助手",
        "subtitle": "专为爷爷定制的血压监测器",
        "add_entry": "新增血压记录",
        "view_summary": "查看摘要（终端）",
        "plot_trend": "绘制趋势图",
        "detect_anomalies": "检测异常",
        "forecast": "LSTM 预测",
        "report": "生成完整报告（PDF）",
        "exit": "退出程序",
        "form_title": "输入爷爷的血压记录",
        "form_labels": ["日期 (YYYY-MM-DD)", "时间 (HH:MM)", "收缩压", "舒张压", "心率", "备注"],
        "form_button": "保存记录",
        "saved": "记录保存成功！",
        "error_numbers": "收缩压、舒张压和心率必须为整数。",
        "report_done": "报告已保存为 bp_full_report.pdf",
        "lang_toggle": "EN"
    }
}

current_lang = "en"


def run_ui():
    global current_lang
    lang = LANGUAGES[current_lang]

    root = tk.Tk()
    root.title(lang["title"])
    root.geometry("460x580")
    root.configure(bg="#f0f2f5")

    def reload_ui():
        root.destroy()
        run_ui()

    # Header
    title_frame = tk.Frame(root, bg="#f0f2f5")
    title_frame.pack(pady=20)
    tk.Label(title_frame, text=lang["title"], font=("Helvetica", 26, "bold"), bg="#f0f2f5", fg="#2b2b2b").pack()
    tk.Label(title_frame, text=lang["subtitle"], font=("Helvetica", 12), bg="#f0f2f5", fg="#555555").pack()

    # Language toggle
    tk.Button(root, text=lang["lang_toggle"], font=("Helvetica", 10), bg="#dddddd", fg="#000000",
              command=lambda: toggle_language(root)).pack(pady=(0, 10))

    # Button builder
    def make_button(label, command):
        return tk.Button(
            root, text=label, command=command,
            font=("Helvetica", 12),
            bg="#4A90E2", fg="white",
            activebackground="#357ABD", activeforeground="white",
            relief="flat", width=32, height=2
        )

    def toggle_language(window):
        global current_lang
        current_lang = "zh" if current_lang == "en" else "en"
        window.destroy()
        run_ui()

    def handle_add_entry():
        form = tk.Toplevel()
        form.title(lang["add_entry"])
        form.geometry("370x460")
        form.configure(bg="#ffffff")

        tk.Label(form, text=lang["form_title"], font=("Helvetica", 14, "bold"), bg="#ffffff").pack(pady=10)
        entries = []
        defaults = [datetime.today().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M'), "", "", "", ""]
        for i, label in enumerate(lang["form_labels"]):
            tk.Label(form, text=label, font=("Helvetica", 10), bg="#ffffff", anchor="w").pack(pady=(10, 0), padx=20, anchor="w")
            e = tk.Entry(form, font=("Helvetica", 11), width=30, bd=1, relief="solid")
            e.insert(0, defaults[i])
            e.pack(pady=2)
            entries.append(e)

        def submit():
            values = [e.get().strip() for e in entries]
            try:
                int(values[2])
                int(values[3])
                int(values[4])
                save_entry(values)
                messagebox.showinfo("Success", lang["saved"])
                form.destroy()
            except ValueError:
                messagebox.showerror("Error", lang["error_numbers"])

        tk.Button(form, text=lang["form_button"], command=submit,
                  font=("Helvetica", 12), bg="#4A90E2", fg="white",
                  activebackground="#357ABD", activeforeground="white",
                  relief="flat", width=20).pack(pady=25)

    def handle_summary():      run_callback(view_summary)
    def handle_plot():         run_callback(plot_readings)
    def handle_anomaly():      run_callback(detect_anomalies)
    def handle_forecast():     run_callback(train_lstm_model)
    def handle_report():
        try:
            generate_report()
            messagebox.showinfo("Done", lang["report_done"])
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_callback(func):
        try:
            func()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Buttons
    actions = [
        (lang["add_entry"], handle_add_entry),
        (lang["view_summary"], handle_summary),
        (lang["plot_trend"], handle_plot),
        (lang["detect_anomalies"], handle_anomaly),
        (lang["forecast"], handle_forecast),
        (lang["report"], handle_report),
        (lang["exit"], root.destroy)
    ]

    for label, callback in actions:
        make_button(label, callback).pack(pady=6)

    root.mainloop()

if __name__ == "__main__":
    run_ui()