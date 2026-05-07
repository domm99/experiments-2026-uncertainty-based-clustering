var prepareCmd = `
echo VERSION="\${nextRelease.version}" > .env
echo PROJECT_NAME= 'Experiments on Neighbor-based Transfer Learning in MARL' >> .env
docker build -t davidedomini99/experiments-2025-tcd-marl-neighbor-tl:\${nextRelease.version} .
`
var publishCmd = `
docker push davidedomini99/experiments-2025-tcd-marl-neighbor-tl:\${nextRelease.version}
git add .env
git commit -m "chore(release): update .env versions to \${nextRelease.version} [skip ci]"
git push
`
var config = require('semantic-release-preconfigured-conventional-commits');
config.plugins.push(
    ["@semantic-release/exec", {
        "prepareCmd": prepareCmd,
        "publishCmd": publishCmd,
    }],
    ["@semantic-release/github", {
        "assets": [
            { "path": "charts.tar.zst" },
        ]
    }],
    "@semantic-release/git",
)
module.exports = config