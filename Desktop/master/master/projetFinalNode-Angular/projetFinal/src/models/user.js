module.exports = (sequelize, DataTypes) => {
    return sequelize.define('User', {
      id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        autoIncrement: true
      },
      username: {
        type: DataTypes.STRING,
        unique:{msg:'le nom d_utilisteur est deja pris'}
      },
      password: {
        type: DataTypes.STRING
      },
      isAdmin:{
       type:DataTypes.STRING,
      }
     
      

    })
  }