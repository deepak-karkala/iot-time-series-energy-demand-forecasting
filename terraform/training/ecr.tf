resource "aws_ecr_repository" "edf_training_repo" {
  name                 = local.ecr_repo_name_edf_full
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration { scan_on_push = true }
  tags = local.tags
}