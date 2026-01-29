Dưới đây là công thức toán học chi tiết của Edge-aware Smoothness Loss mà tôi đề xuất, được biểu diễn từ đoạn code C++ sang ngôn ngữ Toán học.

Công thức tổng quát
Hàm loss tổng thể ($L_{smooth}$) là trung bình cộng của loss theo phương ngang ($x$) và phương dọc ($y$):

$$L_{smooth} = \frac{1}{N} \sum_{p} \left( \lambda_x(p) \cdot |\partial_x D(p)| + \lambda_y(p) \cdot |\partial_y D(p)| \right)$$

Trong đó:

$p$: Vị trí pixel $(x, y)$ trên ảnh.
$N$: Tổng số pixel.
$D(p)$: Giá trị độ sâu (Depth) tại pixel $p$.
$I(p)$: Giá trị màu (Image) tại pixel $p$ (đã chuẩn hóa về 0-1).
Giải thích từng thành phần
1. Độ thay đổi của Depth ($|\partial D|$) - "Muốn phẳng"
Đây là thành phần khởi nguồn của lực "kéo". Nó đo độ chênh lệch độ sâu giữa 2 pixel kề nhau.

Theo trục x: $|\partial_x D(x, y)| = |D(x+1, y) - D(x, y)|$
Theo trục y: $|\partial_y D(x, y)| = |D(x, y+1) - D(x, y)|$
$\rightarrow$ Nếu giá trị này lớn, tức là bề mặt đang gồ ghề (pixel này lồi lên hoặc lõm xuống so với pixel bên cạnh). Loss sẽ muốn ép giá trị này về 0.

2. Trọng số từ ảnh RGB ($\lambda$) - "Biết nhìn cạnh"
Đây là thành phần điều khiển lực kéo. Nó quyết định xem "tại điểm này có nên làm phẳng hay không?".

Công thức tính trọng số (weight): $$\lambda_x(p) = e^{-|\partial_x I(p)|}$$ $$\lambda_y(p) = e^{-|\partial_y I(p)|}$$

Trong đó $|\partial_x I|$ là gradient của ảnh màu (độ thay đổi màu sắc): $$|\partial_x I(p)| = \text{mean}\big(|I_{R,G,B}(x+1, y) - I_{R,G,B}(x, y)|\big)$$

Cơ chế hoạt động của hàm $e^{-x}$:

Vùng phẳng (Tường trắng): Màu pixel $A$ giống màu pixel $B$ $\rightarrow$ Gradient $|\partial I| \approx 0$.
Trọng số $\lambda = e^0 = \mathbf{1}$.
Kết quả: Lực làm phẳng tác động mạnh nhất (100% công lực). Gaussian bị kéo mạnh để nằm cùng mặt phẳng.
Vùng cạnh (Góc bàn, mép vật thể): Màu pixel thay đổi đột ngột $\rightarrow$ Gradient $|\partial I|$ rất lớn (ví dụ = 5).
Trọng số $\lambda = e^{-5} \approx \mathbf{0.006}$.
Kết quả: Lực làm phẳng gần như biến mất. Loss "tha" cho chỗ này, cho phép Depth thay đổi đột ngột (để tạo ra cạnh sắc nét của vật thể) mà không bị làm mờ.
Tóm lại ý nghĩa vật lý
Công thức này nói với máy tính rằng:

"Hãy san phẳng độ sâu ở những nơi màu sắc đồng nhất (tường, sàn), nhưng nếu thấy màu thay đổi đột ngột (cạnh vật thể) thì đừng can thiệp."