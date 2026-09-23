"""Validate local asset links and prepare a clean GitHub Pages source ZIP."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []
        self.times = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            assert attrs['id'] not in self.ids, f'Duplicate id: {attrs["id"]}'
            self.ids.add(attrs['id'])
        for field in ['src', 'href', 'poster']:
            if field in attrs:
                self.links.append(attrs[field])
        if 'data-time' in attrs:
            self.times.append(int(attrs['data-time']))
        if tag == 'img':
            assert attrs.get('alt'), 'Image needs alternative text'


page = Page()
page.feed((ROOT / 'index.html').read_text(encoding='utf-8'))
assert page.times == [0, 8, 28, 50], 'Video chapters differ from renderer'
for link in page.links:
    url = urlsplit(link)
    if url.scheme or url.netloc:
        continue
    if not url.path:
        assert not url.fragment or url.fragment in page.ids, f'Missing anchor: {link}'
    else:
        path = (ROOT / unquote(url.path)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(), f'Missing local asset: {link}'

paths = [ROOT / name for name in ['index.html', '.nojekyll',
         '.gitignore', '.gitattributes']]
# Ship referenced assets only; paper PDFs are never deployment assets.
assets = set()
for link in page.links:
    url = urlsplit(link)
    if not url.scheme and not url.netloc and url.path:
        asset = (ROOT / unquote(url.path)).resolve()
        assert asset.suffix.lower() != '.pdf', 'The project page must not link to a PDF'
        if asset.is_relative_to(ROOT / 'static'):
            assets.add(asset)
assets.add(ROOT / 'static' / 'images' / 'social-preview.png')
paths += sorted(assets)
paths += [ROOT / 'scripts' / name for name in
          ['render_animation.py', 'requirements-animation.txt', 'package_site.py']]
target = ROOT / 'dist' / 'latchq-github-pages.zip'
target.parent.mkdir(exist_ok=True)
with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
    for path in paths:
        if path.is_file():
            archive.write(path, path.relative_to(ROOT).as_posix())
with ZipFile(target) as archive:
    assert archive.testzip() is None
    assert not any(name.lower().endswith('.pdf') for name in archive.namelist()), 'PDF found in deployment ZIP'
    assert not any(name.startswith(('work/', 'reference-nerfies/', 'site/', '.git/', '.local-backup/'))
                   for name in archive.namelist())
    print(f'Validated {len(page.links)} links and four video chapters.')
    print(f'Packaged {len(archive.namelist())} files: {target.name} ({target.stat().st_size:,} bytes).')
