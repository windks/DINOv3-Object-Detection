#!/usr/bin/env python3
"""
DINOv3 Object Detection - GUI 버전
마우스로 쉽게 사용할 수 있는 그래픽 인터페이스

사용법:
    python detect_gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
from pathlib import Path
import json

from production_api import DINOv3DetectorAPI


class DINOv3DetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DINOv3 Object Detection - GUI")
        self.root.geometry("1200x800")
        
        # 상태 변수
        self.detector = None
        self.current_image = None
        self.current_results = None
        self.image_path = None
        
        # UI 구성
        self.setup_ui()
        
        # 모델 로드 (백그라운드)
        self.load_model_async()
    
    def setup_ui(self):
        """UI 구성"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 상단 컨트롤 패널
        control_frame = ttk.LabelFrame(main_frame, text="컨트롤", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 이미지 선택
        ttk.Button(control_frame, text="이미지 열기", command=self.load_image).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="폴더 열기", command=self.load_folder).grid(row=0, column=1, padx=5)
        
        # 모델 선택
        ttk.Label(control_frame, text="모델:").grid(row=0, column=2, padx=(20, 5))
        self.model_var = tk.StringVar(value="small")
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                  values=["small", "base", "large"], width=10)
        model_combo.grid(row=0, column=3, padx=5)
        model_combo.bind("<<ComboboxSelected>>", self.on_model_change)
        
        # 임계값
        ttk.Label(control_frame, text="신뢰도:").grid(row=0, column=4, padx=(20, 5))
        self.threshold_var = tk.DoubleVar(value=0.3)
        threshold_scale = ttk.Scale(control_frame, from_=0.1, to=0.9, 
                                   variable=self.threshold_var, orient=tk.HORIZONTAL, length=100)
        threshold_scale.grid(row=0, column=5, padx=5)
        self.threshold_label = ttk.Label(control_frame, text="0.30")
        self.threshold_label.grid(row=0, column=6, padx=5)
        threshold_scale.configure(command=self.update_threshold_label)
        
        # 타겟 입력
        target_frame = ttk.LabelFrame(main_frame, text="찾을 객체 (콤마로 구분)", padding="10")
        target_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.target_entry = ttk.Entry(target_frame, width=80)
        self.target_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.target_entry.insert(0, "사람, 자동차, 강아지, 고양이, 자전거")
        
        ttk.Button(target_frame, text="탐지 실행", command=self.detect_objects).grid(row=0, column=1)
        
        # 빠른 선택 버튼
        quick_frame = ttk.Frame(target_frame)
        quick_frame.grid(row=1, column=0, columnspan=2, pady=(5, 0))
        
        quick_targets = [
            ("일반", "사람, 자동차, 자전거, 개, 고양이"),
            ("교통", "자동차, 트럭, 버스, 오토바이, 신호등"),
            ("동물", "개, 고양이, 새, 말, 소"),
            ("실내", "의자, 테이블, 노트북, 휴대폰, 컵")
        ]
        
        for i, (name, targets) in enumerate(quick_targets):
            ttk.Button(quick_frame, text=name, 
                      command=lambda t=targets: self.set_targets(t)).grid(row=0, column=i, padx=2)
        
        # 이미지 표시 영역
        image_frame = ttk.LabelFrame(main_frame, text="이미지", padding="10")
        image_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.canvas = tk.Canvas(image_frame, width=600, height=400, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 결과 표시 영역
        result_frame = ttk.LabelFrame(main_frame, text="탐지 결과", padding="10")
        result_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.result_text = scrolledtext.ScrolledText(result_frame, width=40, height=25)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # 하단 버튼
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(button_frame, text="결과 저장", command=self.save_result).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="JSON 내보내기", command=self.export_json).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="CSV 내보내기", command=self.export_csv).grid(row=0, column=2, padx=5)
        
        # 상태바
        self.status_var = tk.StringVar(value="준비 중...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 가중치 설정
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def load_model_async(self):
        """백그라운드에서 모델 로드"""
        def load():
            self.status_var.set("모델 로딩 중... (첫 실행시 다운로드로 시간이 걸립니다)")
            try:
                self.detector = DINOv3DetectorAPI(model_size=self.model_var.get())
                self.status_var.set("준비 완료!")
            except Exception as e:
                self.status_var.set(f"모델 로드 실패: {str(e)}")
                messagebox.showerror("오류", f"모델 로드 실패:\n{str(e)}")
        
        threading.Thread(target=load, daemon=True).start()
    
    def on_model_change(self, event=None):
        """모델 변경"""
        if messagebox.askyesno("모델 변경", "모델을 변경하시겠습니까?\n(다시 로드하는데 시간이 걸립니다)"):
            self.load_model_async()
    
    def update_threshold_label(self, value):
        """임계값 레이블 업데이트"""
        self.threshold_label.config(text=f"{float(value):.2f}")
    
    def set_targets(self, targets):
        """타겟 설정"""
        self.target_entry.delete(0, tk.END)
        self.target_entry.insert(0, targets)
    
    def load_image(self):
        """이미지 파일 열기"""
        file_path = filedialog.askopenfilename(
            title="이미지 선택",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.status_var.set(f"이미지 로드: {Path(file_path).name}")
    
    def load_folder(self):
        """폴더 선택 (추후 구현)"""
        messagebox.showinfo("정보", "폴더 일괄 처리는 detect_folder.py를 사용하세요.")
    
    def display_image(self, image_path):
        """이미지 표시"""
        # 이미지 로드
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image = image_rgb
        
        # 캔버스 크기에 맞게 리사이즈
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            image_pil = Image.fromarray(image_rgb)
            image_pil.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            # tkinter 이미지로 변환
            photo = ImageTk.PhotoImage(image_pil)
            
            # 캔버스에 표시
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
            self.canvas.image = photo  # 참조 유지
    
    def detect_objects(self):
        """객체 탐지 실행"""
        if not self.detector:
            messagebox.showerror("오류", "모델이 아직 로드되지 않았습니다.")
            return
        
        if self.current_image is None:
            messagebox.showerror("오류", "먼저 이미지를 선택하세요.")
            return
        
        # 타겟 파싱
        targets = [t.strip() for t in self.target_entry.get().split(",") if t.strip()]
        
        if not targets:
            messagebox.showerror("오류", "찾을 객체를 입력하세요.")
            return
        
        self.status_var.set("탐지 중...")
        self.result_text.delete(1.0, tk.END)
        
        # 탐지 실행 (백그라운드)
        def detect():
            try:
                results = self.detector.detect(
                    self.current_image,
                    targets,
                    confidence_threshold=self.threshold_var.get()
                )
                
                self.current_results = results
                
                # 결과 표시
                self.display_results(results)
                
                # 시각화
                vis_image = self.detector.visualize(self.current_image, results)
                self.display_visualization(vis_image)
                
                self.status_var.set(f"탐지 완료: {len(results)}개 객체 발견")
                
            except Exception as e:
                self.status_var.set(f"탐지 실패: {str(e)}")
                messagebox.showerror("오류", f"탐지 실패:\n{str(e)}")
        
        threading.Thread(target=detect, daemon=True).start()
    
    def display_results(self, results):
        """결과 텍스트 표시"""
        self.result_text.delete(1.0, tk.END)
        
        if not results:
            self.result_text.insert(tk.END, "탐지된 객체가 없습니다.\n\n")
            self.result_text.insert(tk.END, "팁:\n")
            self.result_text.insert(tk.END, "- 신뢰도 임계값을 낮춰보세요\n")
            self.result_text.insert(tk.END, "- 더 구체적인 설명을 사용하세요\n")
            self.result_text.insert(tk.END, "  예: '차' → '빨간색 자동차'\n")
        else:
            self.result_text.insert(tk.END, f"총 {len(results)}개 객체 탐지\n")
            self.result_text.insert(tk.END, "=" * 40 + "\n\n")
            
            for i, det in enumerate(results, 1):
                self.result_text.insert(tk.END, f"{i}. {det.class_name}\n")
                self.result_text.insert(tk.END, f"   신뢰도: {det.confidence:.1%}\n")
                self.result_text.insert(tk.END, f"   위치: {det.bbox}\n")
                self.result_text.insert(tk.END, f"   크기: {det.bbox[2]-det.bbox[0]}×{det.bbox[3]-det.bbox[1]}\n")
                self.result_text.insert(tk.END, "\n")
    
    def display_visualization(self, vis_image):
        """시각화 이미지 표시"""
        # PIL 이미지로 변환
        image_pil = Image.fromarray(vis_image)
        
        # 캔버스 크기에 맞게 리사이즈
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            image_pil.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            # tkinter 이미지로 변환
            photo = ImageTk.PhotoImage(image_pil)
            
            # 캔버스에 표시
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
            self.canvas.image = photo  # 참조 유지
    
    def save_result(self):
        """결과 이미지 저장"""
        if self.current_results is None:
            messagebox.showwarning("경고", "저장할 결과가 없습니다.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="결과 저장",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            vis_image = self.detector.visualize(self.current_image, self.current_results)
            cv2.imwrite(file_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            self.status_var.set(f"저장 완료: {Path(file_path).name}")
    
    def export_json(self):
        """JSON으로 내보내기"""
        if self.current_results is None:
            messagebox.showwarning("경고", "내보낼 결과가 없습니다.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="JSON 내보내기",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            data = {
                "image": self.image_path,
                "detections": self.detector.to_dict(self.current_results),
                "total": len(self.current_results)
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.status_var.set(f"JSON 내보내기 완료: {Path(file_path).name}")
    
    def export_csv(self):
        """CSV로 내보내기"""
        if self.current_results is None:
            messagebox.showwarning("경고", "내보낼 결과가 없습니다.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="CSV 내보내기",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            import csv
            
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Object', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
                
                for det in self.current_results:
                    writer.writerow([det.class_name, f"{det.confidence:.3f}", *det.bbox])
            
            self.status_var.set(f"CSV 내보내기 완료: {Path(file_path).name}")


def main():
    root = tk.Tk()
    app = DINOv3DetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()