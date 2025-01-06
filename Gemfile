source "https://rubygems.org"

# Use github-pages gem for GitHub Pages compatibility
gem "github-pages", group: :jekyll_plugins

# Jekyll plugins
group :jekyll_plugins do
  gem "jekyll-scholar"
  gem "jekyll-sitemap"
  gem "jekyll-feed"
  gem "jekyll-katex"
  gem "jekyll-seo-tag"
end

# Windows and JRuby dependencies
install_if -> { RUBY_PLATFORM =~ %r!mingw|mswin|java! } do
  gem "tzinfo", "~> 1.2"
  gem "tzinfo-data"
end

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.1", :install_if => Gem.win_platform?