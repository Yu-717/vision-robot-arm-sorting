"""E6 未知物体图片存档：serve.safe_under 的路径边界。

图片名来自 unknown_object 事件的 image 字段，属于外部输入；这个目录又正好是"按名字取文件"，
一旦放行了 ../ 或绝对路径就是任意文件读取。所以这些用例不是凑数，是这条边界的定义。
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'web'))
import serve as S  # noqa: E402


class SafeUnderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'archive'
        self.root.mkdir()
        (self.root / 'a.jpg').write_bytes(b'\xff\xd8\xff')
        (self.root / 'day').mkdir()
        (self.root / 'day' / 'b.png').write_bytes(b'\x89PNG')

    def ok(self, name):
        got = S.safe_under(self.root, name)
        self.assertIsNotNone(got, f'{name!r} 应当被放行')
        return got

    def refused(self, name):
        self.assertIsNone(S.safe_under(self.root, name), f'{name!r} 必须被拒绝')

    def test_a_plain_name_in_the_directory_is_allowed(self):
        self.assertEqual(self.ok('a.jpg'), self.root / 'a.jpg')

    def test_a_name_in_a_subdirectory_is_allowed(self):
        self.assertEqual(self.ok('day/b.png'), self.root / 'day' / 'b.png')

    def test_parent_traversal_is_refused(self):
        self.refused('../outside.jpg')
        self.refused('day/../../outside.jpg')
        self.refused('..')

    def test_absolute_paths_are_refused(self):
        # root / 绝对路径 会被 join 成那个绝对路径本身，必须靠"是否在 root 内"拦下来
        self.refused('/etc/passwd')
        self.refused(str(self.root.parent / 'outside.jpg'))

    def test_windows_drive_letters_are_refused(self):
        self.refused('C:\\Windows\\win.ini')

    def test_empty_and_nul_names_are_refused(self):
        self.refused('')
        self.refused('a\x00.jpg')

    def test_the_directory_itself_is_not_a_file_to_serve(self):
        self.refused('.')


if __name__ == '__main__':
    unittest.main()
