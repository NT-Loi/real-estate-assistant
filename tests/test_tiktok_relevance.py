import unittest

from crawlers.neighborhood.tiktok import TikTokCrawler


class TikTokRelevanceTest(unittest.TestCase):
    def test_rejects_unrelated_vehicle_video(self):
        keyword = "review Vlasta Premier Phú Thuận, Đường Đào Trí, Phường Phú Thuận, Hồ Chí Minh"
        title = 'POV: Bạn tìm thấy "chân ái" trong thế giới xe điện #wuling'

        self.assertFalse(TikTokCrawler.is_relevant_to_keyword(keyword, title))

    def test_accepts_matching_project_video(self):
        keyword = "review Vlasta Premier Phú Thuận, Đường Đào Trí, Phường Phú Thuận, Hồ Chí Minh"
        title = "Có Nên Mua Vlasta Premier Phú Thuận? Review tổ hợp căn hộ resort 5 sao"

        self.assertTrue(TikTokCrawler.is_relevant_to_keyword(keyword, title))

    def test_accepts_strong_single_brand_match(self):
        keyword = "review Vinhomes Green Paradise, Xã Cần Giờ, Hồ Chí Minh"
        title = "Vinhomes Cần Giờ cập nhật tiến độ mới nhất"

        self.assertTrue(TikTokCrawler.is_relevant_to_keyword(keyword, title))


if __name__ == "__main__":
    unittest.main()
