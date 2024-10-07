import cv2 as cv
import os
import time

# ฟังก์ชันเพื่อรับข้อมูลสมาชิก
def get_members_info():
    members_info = []
    num_members = int(input("Enter the number of members: "))
    
    for _ in range(num_members):
        name = input("Enter the member's name: ")
        members_info.append(name)  # เก็บแค่ชื่อสมาชิก
        
    return members_info

# ฟังก์ชันนับถอยหลังโดยที่ภาพไม่ค้าง
def countdown(cam):
    start_time = time.time()  # เริ่มนับเวลา
    countdown_duration = 3  # ระยะเวลานับถอยหลัง (3 วินาที)
    while True:
        elapsed_time = int(time.time() - start_time)
        remaining_time = countdown_duration - elapsed_time
        
        check, frame = cam.read()
        if check:
            frame = cv.flip(frame, 1)

            # แสดงตัวเลขนับถอยหลังตรงกลางจอ
            if remaining_time > 0:
                cv.putText(frame, str(remaining_time), (frame.shape[1] // 2 - 50, frame.shape[0] // 2), 
                           cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
            else:
                # ถ้านับถอยหลังเสร็จแล้ว ให้หยุดลูป
                break

            cv.imshow("Output", frame)

            if cv.waitKey(1) & 0xFF == ord('e'):  # กด 'e' เพื่อออก
                cam.release()
                cv.destroyAllWindows()
                exit(0)
        else:
            break

# รับข้อมูลสมาชิก
members_info = get_members_info()

cam = cv.VideoCapture(0)

cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

image_counter = 1
frame_count = 0
frame_save = 15  # ลดจำนวนเฟรมที่ใช้บันทึกภาพให้เร็วขึ้น

# จำนวนรูปต่อคน
images_per_member = 200

# สร้างโฟลเดอร์ output หากยังไม่มี
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

member_index = 0

cv.namedWindow("Output", cv.WINDOW_NORMAL)
cv.setWindowProperty("Output", cv.WND_PROP_TOPMOST, 1)

# ตัวแปรสถานะ pause
is_paused = False

# นับถอยหลังก่อนเริ่มถ่ายรูปครั้งแรก
countdown(cam)

while True:
    check, frame = cam.read()
    if check:
        # Flip the image horizontally
        frame = cv.flip(frame, 1)
        
        # แปลงภาพเป็นขาวดำเพื่อใช้ในการตรวจจับใบหน้า
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # ตรวจจับใบหน้า
        detect = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)  # ปรับค่า scaleFactor และ minNeighbors
        
        # ทำสำเนาของ frame ก่อนวาดกรอบ
        original_frame = frame.copy()

        # วาดกรอบสี่เหลี่ยมรอบใบหน้า และบันทึกภาพเมื่อเจอใบหน้า
        for (x, y, w, h) in detect:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # แสดงข้อความ "Photo: x" ที่มุมขวาบน
            cv.putText(frame, f"Photo: {image_counter - 1}", 
                       (frame.shape[1] - 150, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if not is_paused and frame_count % frame_save == 0:
                if image_counter <= images_per_member:
                    # สร้างโฟลเดอร์สำหรับสมาชิกหากยังไม่มี
                    member_folder = os.path.join(output_folder, members_info[member_index])
                    if not os.path.exists(member_folder):
                        os.makedirs(member_folder)

                    # ครอบเฉพาะใบหน้าจาก original_frame ที่ไม่มีกรอบ
                    face_crop = original_frame[y:y+h, x:x+w]
                    
                    # ตั้งชื่อไฟล์รูปภาพให้มีแค่การนับเฟรม
                    img_name = f"{image_counter}.jpg"  # เปลี่ยนชื่อไฟล์เป็นแค่หมายเลข
                    img_path = os.path.join(member_folder, img_name)
                    cv.imwrite(img_path, face_crop)
                    print(f"{img_name} Save complete!")
                    image_counter += 1
                
                # หากถ่ายครบจำนวนที่ตั้งไว้แล้ว เปลี่ยนไปถ่ายสมาชิกคนถัดไป
                if image_counter > images_per_member:
                    print(f"Completed taking {images_per_member} photos for {members_info[member_index]}")
                    
                    # เช็คว่าถึงคนสุดท้ายหรือยัง
                    if member_index < len(members_info) - 1:
                        print("Press 'n' to switch to the next member, or 'e' to exit")
                    else:
                        print("Completed taking photos of everyone!")

                    # รอการกดปุ่ม n หรือ e โดยยังแสดงภาพจากกล้อง
                    while True:
                        check, frame = cam.read()
                        if check:
                            frame = cv.flip(frame, 1)
                            cv.putText(frame, f"Photo: {image_counter - 1}", 
                                       (frame.shape[1] - 150, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv.imshow("Output", frame)

                            # ตรวจสอบการกดปุ่ม
                            key = cv.waitKey(1) & 0xFF
                            if key == ord('n'):
                                member_index += 1
                                image_counter = 1
                                if member_index >= len(members_info):
                                    print("All photos have been taken!")
                                    cam.release()
                                    cv.destroyAllWindows()
                                    exit(0)
                                # นับถอยหลังก่อนถ่ายสมาชิกถัดไป
                                countdown(cam)
                                break
                            elif key == ord('e'):
                                cam.release()
                                cv.destroyAllWindows()
                                exit(0)
                        else:
                            break
        
        # แสดงข้อความ "PAUSE" สีแดงเมื่อ pause
        if is_paused:
            cv.putText(frame, "PAUSED", (frame.shape[1] // 2 - 100, frame.shape[0] // 2), 
                       cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        # แสดงผลเฟรมปัจจุบัน
        cv.imshow("Output", frame)
        
        # เพิ่มค่า frame_counter
        frame_count += 1

        # กด 'p' เพื่อ pause หรือ resume
        key = cv.waitKey(1) & 0xFF
        if key == ord('p'):
            is_paused = not is_paused
            if is_paused:
                print("Paused")
            else:
                print("Resumed")
                # นับถอยหลังก่อน resume
                countdown(cam)
        
        # กด 'e' เพื่อออก
        if key == ord('e'):
            break
    else:
        break

# ปิดกล้องและทำลายหน้าต่างทั้งหมด
cam.release()
cv.destroyAllWindows()
