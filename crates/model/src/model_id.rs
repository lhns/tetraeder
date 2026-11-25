//! Model identity for a cluster (ADR 0091): the name every node keeps a model under.
//!
//! One file: its sha256, kept as `<models>/<id>.gguf`. A split GGUF (llama.cpp's
//! `<prefix>-00001-of-0000N.gguf`, as the 235B ships): the sha256 of a manifest that lists each
//! shard's index, size and sha256 (no file names), kept as `<models>/<id>/manifest.txt` with the
//! shards as `<id>-0000k-of-0000N.gguf` beside it. The id is 64 hex digits either way, so the
//! wire (`model_sha256`) and the path-safety rule (hex only) do not change.
//!
//! A node checks a model without hashing it again: the manifest must hash to the id, and every
//! shard must exist with the size the manifest lists (hashing 142 GB at every start would cost
//! minutes; `identify` hashes once, where the model is installed).

use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

pub const MANIFEST_FILE: &str = "manifest.txt";
const MANIFEST_HEAD: &str = "tetraeder-shards 1";

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

pub fn is_id(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// sha256 (lowercase hex) of the bytes of a file, and its length.
pub fn file_sha256(path: &Path) -> io::Result<(String, u64)> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 4 << 20];
    let mut n = 0u64;
    loop {
        let k = f.read(&mut buf)?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((hex(&h.finalize()), n))
}

fn hex(b: &[u8]) -> String {
    b.iter().fold(String::new(), |mut s, x| {
        let _ = write!(s, "{x:02x}");
        s
    })
}

fn sha256_text(text: &str) -> String {
    hex(&Sha256::digest(text.as_bytes()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shard {
    pub bytes: u64,
    pub sha256: String,
}

/// The shards of one split model, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub shards: Vec<Shard>,
}

impl Manifest {
    /// The canonical text: a header line, then `<no> <bytes> <sha256>` per shard, numbered from 1.
    pub fn to_text(&self) -> String {
        let mut s = format!("{MANIFEST_HEAD}\n");
        for (i, sh) in self.shards.iter().enumerate() {
            let _ = writeln!(s, "{} {} {}", i + 1, sh.bytes, sh.sha256);
        }
        s
    }

    pub fn id(&self) -> String {
        sha256_text(&self.to_text())
    }

    pub fn parse(text: &str) -> io::Result<Self> {
        let mut lines = text.lines();
        if lines.next() != Some(MANIFEST_HEAD) {
            return Err(invalid(format!(
                "manifest does not start with `{MANIFEST_HEAD}`"
            )));
        }
        let mut shards = Vec::new();
        for (i, l) in lines.enumerate() {
            let f: Vec<&str> = l.split(' ').collect();
            let [no, bytes, sha] = f[..] else {
                return Err(invalid(format!(
                    "manifest line `{l}` is not `<no> <bytes> <sha256>`"
                )));
            };
            if no.parse::<usize>().ok() != Some(i + 1) || !is_id(sha) {
                return Err(invalid(format!(
                    "manifest line `{l}` is out of order or not a sha256"
                )));
            }
            let bytes = bytes
                .parse()
                .map_err(|_| invalid(format!("manifest line `{l}` has no byte count")))?;
            shards.push(Shard {
                bytes,
                sha256: sha.to_ascii_lowercase(),
            });
        }
        if shards.is_empty() {
            return Err(invalid("manifest lists no shard".into()));
        }
        let m = Self { shards };
        // Canonical text only, so one model has one id.
        if m.to_text() != text {
            return Err(invalid("manifest is not in canonical form".into()));
        }
        Ok(m)
    }
}

/// `<id>-00003-of-00005.gguf`; `no` is 1-based.
pub fn shard_file_name(id: &str, no: usize, count: usize) -> String {
    format!("{id}-{no:05}-of-{count:05}.gguf")
}

/// What `resolve` found: the file to open (the first shard of a split model) and how many shards
/// and bytes the model has.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Located {
    pub path: PathBuf,
    pub shards: usize,
    pub bytes: u64,
}

/// Finds model `id` under `models` and checks it: a single `<id>.gguf`, else the directory `<id>`
/// with its manifest and shards. The manifest must hash to `id` and every shard must have the
/// listed size.
pub fn resolve(models: &Path, id: &str) -> io::Result<Located> {
    // The id names a path: only hex, so it cannot name another one.
    if !is_id(id) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "model_sha256 is not 64 hex digits",
        ));
    }
    let id = id.to_ascii_lowercase();
    let single = models.join(format!("{id}.gguf"));
    if single.is_file() {
        return Ok(Located {
            bytes: std::fs::metadata(&single)?.len(),
            path: single,
            shards: 1,
        });
    }
    let dir = models.join(&id);
    let text = match std::fs::read_to_string(dir.join(MANIFEST_FILE)) {
        Ok(t) => t,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "no model {id}: neither {id}.gguf nor {id}/{MANIFEST_FILE} in {}",
                    models.display()
                ),
            ));
        }
        Err(e) => return Err(e),
    };
    let m = Manifest::parse(&text)?;
    if m.id() != id {
        return Err(invalid(format!(
            "model {id}: its manifest hashes to {}",
            m.id()
        )));
    }
    let count = m.shards.len();
    let mut bytes = 0;
    for (i, sh) in m.shards.iter().enumerate() {
        let p = dir.join(shard_file_name(&id, i + 1, count));
        let len = std::fs::metadata(&p)
            .map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!(
                        "model {id}: shard {} of {count} ({}): {e}",
                        i + 1,
                        p.display()
                    ),
                )
            })?
            .len();
        if len != sh.bytes {
            return Err(invalid(format!(
                "model {id}: shard {} of {count} has {len} bytes, the manifest says {}",
                i + 1,
                sh.bytes
            )));
        }
        bytes += len;
    }
    Ok(Located {
        path: dir.join(shard_file_name(&id, 1, count)),
        shards: count,
        bytes,
    })
}

/// A model on disk, identified.
#[derive(Debug, Clone)]
pub struct Identity {
    pub id: String,
    /// `None` for a single file.
    pub manifest: Option<Manifest>,
    /// The files in shard order (one for a single file).
    pub files: Vec<PathBuf>,
}

/// `Some((prefix, count))` when `name` is the first shard of a split model.
fn first_shard(name: &str) -> Option<(&str, usize)> {
    let stem = name.strip_suffix(".gguf")?;
    let (head, count) = stem.rsplit_once("-of-")?;
    let prefix = head.strip_suffix("-00001")?;
    let n: usize = count.parse().ok().filter(|_| count.len() == 5)?;
    (n >= 1).then_some((prefix, n))
}

/// Hashes the model at `path`: a first shard (`<prefix>-00001-of-0000N.gguf`) names the whole
/// split model, any other file is a single file.
pub fn identify(path: &Path) -> io::Result<Identity> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let Some((prefix, count)) = first_shard(name) else {
        let (id, _) = file_sha256(path)?;
        return Ok(Identity {
            id,
            manifest: None,
            files: vec![path.to_path_buf()],
        });
    };
    let mut files = Vec::new();
    let mut shards = Vec::new();
    for no in 1..=count {
        let p = path.with_file_name(format!("{prefix}-{no:05}-of-{count:05}.gguf"));
        let (sha256, bytes) = file_sha256(&p).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("shard {no} of {count} ({}): {e}", p.display()),
            )
        })?;
        shards.push(Shard { bytes, sha256 });
        files.push(p);
    }
    let m = Manifest { shards };
    Ok(Identity {
        id: m.id(),
        manifest: Some(m),
        files,
    })
}

/// Links the model into `models` under its id (a hard link, else a symbolic link on Unix, never
/// a copy), as `resolve` reads it; returns the path `resolve` will give.
pub fn install(i: &Identity, models: &Path) -> io::Result<PathBuf> {
    let place = |from: &Path, to: &Path| -> io::Result<()> {
        if to.exists() {
            return Ok(());
        }
        // Never a copy: the 235B is 142 GB and the disk of a node 230 GB.
        std::fs::hard_link(from, to).or_else(|e| link_across(from, to, e))
    };
    std::fs::create_dir_all(models)?;
    match &i.manifest {
        None => {
            let to = models.join(format!("{}.gguf", i.id));
            place(&i.files[0], &to)?;
            Ok(to)
        }
        Some(m) => {
            let dir = models.join(&i.id);
            std::fs::create_dir_all(&dir)?;
            let count = m.shards.len();
            for (k, f) in i.files.iter().enumerate() {
                place(f, &dir.join(shard_file_name(&i.id, k + 1, count)))?;
            }
            std::fs::write(dir.join(MANIFEST_FILE), m.to_text())?;
            Ok(dir.join(shard_file_name(&i.id, 1, count)))
        }
    }
}

#[cfg(unix)]
fn link_across(from: &Path, to: &Path, _: io::Error) -> io::Result<()> {
    // Another file system: a symbolic link to the absolute path.
    std::os::unix::fs::symlink(std::fs::canonicalize(from)?, to)
}

#[cfg(not(unix))]
fn link_across(_: &Path, _: &Path, e: io::Error) -> io::Result<()> {
    Err(io::Error::new(
        e.kind(),
        format!(
            "cannot link the model files into the models folder (it must be on the same disk): {e}"
        ),
    ))
}

/// The model id a path stands for: its stem when that is an id (`<id>.gguf`), or the name of the
/// directory that holds a split model (`<id>/<id>-00001-of-0000N.gguf`).
pub fn id_of_path(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    if is_id(stem) {
        return Some(stem.to_ascii_lowercase());
    }
    let dir = path.parent()?.file_name()?.to_str()?;
    (is_id(dir) && first_shard(path.file_name()?.to_str()?).is_some())
        .then(|| dir.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(c: char) -> String {
        c.to_string().repeat(64)
    }

    fn manifest() -> Manifest {
        Manifest {
            shards: vec![
                Shard {
                    bytes: 10,
                    sha256: sha('a'),
                },
                Shard {
                    bytes: 20,
                    sha256: sha('b'),
                },
                Shard {
                    bytes: 30,
                    sha256: sha('c'),
                },
            ],
        }
    }

    /// hardware: none. One model has one id: it follows the shards' sizes and hashes in order and
    /// nothing else, and the text parses back to the same manifest.
    #[test]
    fn the_id_covers_every_shard_in_order() {
        let m = manifest();
        assert!(is_id(&m.id()));
        assert_eq!(Manifest::parse(&m.to_text()).unwrap(), m);
        let mut other = m.clone();
        other.shards[1].sha256 = sha('d');
        assert_ne!(other.id(), m.id());
        let mut sized = m.clone();
        sized.shards[2].bytes += 1;
        assert_ne!(sized.id(), m.id());
        let mut swapped = m.clone();
        swapped.shards.swap(0, 1);
        assert_ne!(swapped.id(), m.id());
        let mut fewer = m.clone();
        fewer.shards.pop();
        assert_ne!(fewer.id(), m.id());
    }

    /// hardware: none. A manifest that is not canonical, out of order or short is refused.
    #[test]
    fn a_malformed_manifest_is_refused() {
        let good = manifest().to_text();
        assert!(Manifest::parse(&good).is_ok());
        assert!(Manifest::parse("").is_err());
        assert!(Manifest::parse(&good.replace("2 20", "3 20")).is_err());
        assert!(Manifest::parse(&format!("{good}\n")).is_err());
        assert!(Manifest::parse(&good.replace(&sha('a'), "abc")).is_err());
        assert!(Manifest::parse(&good.to_uppercase()).is_err());
        assert!(Manifest::parse(&format!("{MANIFEST_HEAD}\n")).is_err());
    }

    /// hardware: none. First shards are recognised by llama.cpp's naming rule only.
    #[test]
    fn first_shards_are_told_by_name() {
        assert_eq!(first_shard("m-00001-of-00005.gguf"), Some(("m", 5)));
        assert_eq!(first_shard("a-b-00001-of-00002.gguf"), Some(("a-b", 2)));
        assert_eq!(first_shard("m-00002-of-00005.gguf"), None);
        assert_eq!(first_shard("m-1-of-5.gguf"), None);
        assert_eq!(first_shard("m.gguf"), None);
        let id = sha('e');
        let dir = PathBuf::from(&id);
        let first = dir.join(shard_file_name(&id, 1, 5));
        assert_eq!(id_of_path(&first), Some(id.clone()));
        assert_eq!(id_of_path(Path::new(&format!("x/{id}.gguf"))), Some(id));
        assert_eq!(id_of_path(Path::new("x/model.gguf")), None);
    }
}
