from fastapi import Depends, HTTPException
from app.core.security import get_current_user  # Đảm bảo rằng hàm get_current_user được định nghĩa ở đây

# =========================
# ADMIN ONLY ACCESS
# =========================
def require_admin(user=Depends(get_current_user)):
    # Kiểm tra xem người dùng có được xác thực hay không
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Kiểm tra vai trò của người dùng, chỉ cho phép admin
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")

    # ✅ Chuẩn hóa dữ liệu người dùng
    user["id"] = str(user["_id"]) if "_id" in user else user.get("id")

    return user